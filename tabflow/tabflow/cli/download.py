import argparse
import logging
from tabflow.utils.s3_utils import download_from_s3
from tabflow.utils.logging_utils import setup_logging

logger = setup_logging(level=logging.INFO)

def main():
    """
    Utility cli to download files or directories from S3.
    """
    parser = argparse.ArgumentParser(
        description="Download files or directories from S3"
    )
    parser.add_argument(
        "--s3_path", 
        required=True, 
        help="S3 path to download (s3://bucket/path)"
    )
    parser.add_argument(
        "--destination_path", 
        required=True, 
        help="Local destination path"
    )
    
    args = parser.parse_args()
    
    try:
        downloaded_path = download_from_s3(args.s3_path, args.destination_path)
        logger.info(f"Successfully downloaded to {downloaded_path}")
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
    