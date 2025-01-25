from google.auth import default
from google.cloud import storage
from datetime import timedelta
import cv2

period = timedelta(days=5)
credentials, _ = default()

client = storage.Client(credentials=credentials)
bucket = client.bucket("mogram-dev")

checkpoints = {}

def upload_image(
        path: str,
        image
    ) -> str | None:
        """..."""
        blob = bucket.blob(f"mass-score/{path}")

        try:
            success, image = cv2.imencode(".jpeg", image)
            if not success:
                raise Exception("Image encoding failed!")

            image = image.tobytes()
            blob.upload_from_string(image, content_type="image/jpeg")

            url = blob.generate_signed_url(
                expiration=period,
                version="v4",
                method="GET",
            )
            return url
        except:
            return None
        

