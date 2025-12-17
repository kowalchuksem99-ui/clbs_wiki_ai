# core/chroma_repo.py
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from core.utils.singletone import singleton
import logging
import threading
import pandas as pd
from bs4 import BeautifulSoup
import re
import os


# Отключаем телеметрию с помощью переменной окружения
os.environ["CHROMADB_NO_TELEMETRY"] = "1"
import shutil

logger = logging.getLogger(__name__)

@singleton
class ChromaRepository:
    """
    A thread-safe repository for managing documents in ChromaDB.
    Implements the Singleton pattern to ensure a single connection instance.
    """

    def __init__(self,
                 collection_name: str | None = None,
                 model_name: str | None = None
                 ):
        try:
            collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "clbs_wiki")
            model_name = model_name or os.getenv(
                "EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            )

            server_url = os.getenv("CHROMA_SERVER_URL", "").rstrip("/")
            if not server_url:
                raise RuntimeError("CHROMA_SERVER_URL is not set")
            tenant = os.getenv("CHROMA_TENANT", "default_tenant")
            database = os.getenv("CHROMA_DATABASE", "default_database")

            self.client = chromadb.HttpClient(
                host=server_url,
                settings=Settings(anonymized_telemetry=False),
                tenant=tenant,
                database=database,
            )

            # создаём embedding_function
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

            # Получаем/создаём коллекцию с клиентской эмбеддинг-функцией
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

            self.lock = threading.Lock()
            logger.info("ChromaRepository initialized: collection='%s', url='%s'", collection_name, server_url)

        except Exception as e:
            logger.exception(f"Failed to initialize ChromaRepository: {e}")
            raise

    def _clean_content(self, content: Any) -> str:
        """
        Cleans content by removing HTML tags, normalizing whitespace,
        and ensuring UTF-8 safe text. Handles NaN/None values.
        """
        if content is None or pd.isna(content):
            return ""

        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                content = content.decode("cp1251", errors="replace")
        else:
            content = str(content)

        # Убираем HTML
        if re.search(r'<[^>]+>', content):
            soup = BeautifulSoup(content, 'html.parser')
            for a in soup.find_all('a'):
                href = (a.get('href') or '').strip()
                text = a.get_text(strip=True)
                replacement = f"{text} ({href})" if href and text else (href or text)
                a.replace_with(replacement)
            content = soup.get_text(separator=' ', strip=True)

        # Нормализуем пробелы
        content = re.sub(r'\s+', ' ', content).strip()

        return content

    def _prepare_document_data(self, document: Dict[str, Any]) -> Optional[tuple]:
        """Prepares document content and metadata for ChromaDB."""
        # Clean and validate content
        cleaned_content = self._clean_content(document.get('content', ''))
        if not cleaned_content.strip():
            logger.warning(f"Skipping document {document.get('id')} with empty content")
            return None

        # Clean title
        title = self._clean_content(document.get('title', ''))

        # Combine title and content for embedding
        processed_content = f"{title} {cleaned_content}".strip()

        # Build metadata
        metadata = {
            'title': title,
            'parent_id': document.get('parent_id', ''),
            'version_id': document.get('version_id', 1),
            'exists': document.get('exists', True),
            'slug': document.get('slug', ''),
            'favorite_id': document.get('favorite_id', 0),
            'is_public': document.get('is_public', True),
            'date': document.get('date', ''),
            'content_length': len(processed_content)
        }

        # Handle tags
        if document.get('tags'):
            if isinstance(document['tags'], list):
                metadata['tags'] = ",".join(document['tags'])
            else:
                metadata['tags'] = str(document['tags'])

        # Filter out unsupported metadata types
        final_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                final_metadata[k] = v
            else:
                logger.warning(f"Metadata field '{k}' with value '{v}' of type {type(v)} is omitted")

        return processed_content, final_metadata

    def upsert_document(self, document: Dict[str, Any]):
        """
        Atomically creates or updates a document in the collection.
        This is safer than a separate delete and add.

        Args:
            document (Dict[str, Any]): A dictionary representing the document.
        """
        # Clean content before processing
        if 'content' in document:
            document['content'] = self._clean_content(document['content'])
        if 'title' in document:
            document['title'] = self._clean_content(document['title'])

        prepared_data = self._prepare_document_data(document)
        if prepared_data is None:
            return

        processed_content, metadata = prepared_data
        try:
            with self.lock:
                self.collection.upsert(
                    ids=[document['id']],
                    documents=[processed_content],
                    metadatas=[metadata]
                )
            logger.info(f"Upserted document with ID: {document['id']}")
        except Exception as e:
            logger.error(f"Error upserting document with ID {document['id']}: {e}", exc_info=True)
            raise

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a document by its ID.

        Args:
            doc_id (str): The ID of the document.

        Returns:
            Optional[Dict[str, Any]]: The document data or None if not found.
        """
        try:
            with self.lock:
                result = self.collection.get(ids=[doc_id], include=['documents', 'metadatas'])
                if result and result.get('ids'):
                    logger.info(f"Retrieved document with ID: {doc_id}")
                else:
                    logger.info(f"Document with ID: {doc_id} not found.")
                return result
        except Exception as e:
            logger.error(f"Error getting document by ID {doc_id}: {e}", exc_info=True)
            return None

    def find_similar(self, query: str, include: List[str], n_results: int):
        """
        Finds similar documents to a given query.

        Args:
            query (str): The query text.
            include (List[str]): A list of what to include in the results (e.g., 'documents').
            n_results (int): The number of results to return.

        Returns:
            Dict[str, Any]: The query results.
        """
        include = include or ["documents", "metadatas", "distances"]
        with self.lock:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=include
            )
        found = len((results.get("ids") or [[]])[0])
        logger.info(f"Found {found} similar documents for query: '{query}'")
        return results

    def delete(self, doc_id: str) -> bool:
        """
        Deletes a document by its ID.

        Args:
            doc_id (str): The ID of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            with self.lock:
                self.collection.delete(ids=[doc_id])
                logger.info(f"Deleted document with ID: {doc_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
            return False

    def count(self) -> int:
        """
        Counts the total number of documents in the collection.

        Returns:
            int: The total number of documents.
        """
        try:
            with self.lock:
                count = self.collection.count()
                logger.debug(f"Current document count: {count}")
                return count
        except Exception as e:
            logger.error(f"Error counting documents in collection: {e}", exc_info=True)
            return 0

    def import_from_excel(self, file_path: str, batch_size: int = 100) -> int:
        """
        Imports documents from an Excel file into the ChromaDB collection.

        Args:
            file_path (str): Path to the Excel file
            batch_size (int): Number of documents to process in a single batch

        Returns:
            int: Number of successfully imported documents
        """
        try:
            # Чтение Excel-файла
            df = pd.read_excel(file_path, engine='openpyxl')
            logger.info(f"Loaded Excel file with {len(df)} rows")

            # Подготовка данных для пакетной вставки
            documents = []
            metadatas = []
            ids = []
            skipped = 0

            for _, row in df.iterrows():
                raw_content = row.get('Content', '')
                raw_html = row.get('Content_html', '')

                if str(raw_html).strip():
                    final_content = self._clean_content(raw_html)
                else:
                    final_content = self._clean_content(raw_content)

                # Преобразование строки в словарь документа
                doc_dict = {
                    'id': str(row.get('ID', '')),
                    'parent_id': str(row.get('ParentID', '')),
                    'title': self._clean_content(row.get('Title', '')),
                    'content': final_content,
                    'version_id': int(row.get('VersionId', 1)),
                    'exists': bool(row.get('Exists', True)),
                    'versions': self._parse_field(row.get('Versions', [])),
                    'links': self._parse_field(row.get('Links', [])),
                    'views_count': int(row.get('ViewsCount', 0)),
                    'slug': str(row.get('Slug', '')),
                    'favorite_id': int(row.get('FavoriteId', 0)),
                    'is_public': bool(row.get('IsPublic', True)),
                    'tags': self._parse_tags(row.get('Tags', '')),
                    'user_permissions': self._parse_field(row.get('UserPermissions', [])),
                    'date': self._parse_date(row.get('Date', '')),
                }

                # Пропускаем документы без ID
                if not doc_dict['id'] or doc_dict['id'] == 'nan':
                    logger.warning(f"Skipping row with empty ID")
                    skipped += 1
                    continue

                # Подготовка данных для ChromaDB
                prepared_data = self._prepare_document_data(doc_dict)
                if prepared_data is None:
                    skipped += 1
                    continue

                content, metadata = prepared_data
                documents.append(content)
                metadatas.append(metadata)
                ids.append(doc_dict['id'])

            # Пакетная вставка документов
            total = len(ids)
            if total == 0:
                logger.warning("No valid documents found for import")
                return 0

            for i in range(0, total, batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]

                with self.lock:
                    self.collection.upsert(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas
                    )
                logger.info(
                    f"Imported batch {i // batch_size + 1}/{(total - 1) // batch_size + 1} - {len(batch_ids)} documents")

            logger.info(f"Successfully imported {total} documents from Excel (skipped {skipped})")
            return total

        except Exception as e:
            logger.exception(f"Failed to import from Excel: {e}")
            raise

    def _parse_tags(self, tags) -> list:
        """Parses tags from various formats"""
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags if str(tag).strip()]

        if isinstance(tags, str):
            # Handle different formats: string, list in string
            if tags.startswith('[') and tags.endswith(']'):
                try:
                    return [tag.strip(" '\"") for tag in tags[1:-1].split(',')]
                except:
                    pass
            return [tag.strip() for tag in tags.split(',') if tag.strip()]

        return []

    def _parse_field(self, field) -> list:
        """Parses complex fields from Excel"""
        if isinstance(field, list):
            return field

        if isinstance(field, str) and field.startswith('[') and field.endswith(']'):
            try:
                # Safe evaluation for simple lists
                if not any(char in field for char in '{:}'):
                    return [item.strip(" '\"") for item in field[1:-1].split(',')]
            except:
                pass
        return []

    def _parse_date(self, date_str) -> str:
        """Parses various date formats including Excel and /Date(...)/"""
        try:
            # Handle /Date(...)/ format
            if isinstance(date_str, str) and date_str.startswith('/Date('):
                match = re.search(r'\d+', date_str)
                if match:
                    timestamp = int(match.group()) // 1000
                    return pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

            # Handle Excel numeric dates
            if isinstance(date_str, (int, float)):
                return pd.to_datetime(date_str, unit='D', origin='1899-12-30').strftime('%Y-%m-%d %H:%M:%S')

            return str(date_str)
        except:
            return str(date_str)


if __name__ == '__main__':
    # Initialize repository with reset option
    repo = ChromaRepository(
        repository_path='../../data/chroma_db'
    )

    # Import from Excel
    imported_count = repo.import_from_excel(file_path='../../data/wiki_data.xlsx')
    print(f"Imported {imported_count} documents")

    # Print total count
    print(f"Total documents in DB: {repo.count()}")
