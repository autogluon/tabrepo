"""
This is a file for any Constants used throughout tabflow.
Currently we have the docker image aliases as the only constants.
This is done is order to help the user have multiple docker images,
in case they want to run different setups
"""

# Docker image aliases map friendly names to actual URIs
# Sample usage, make sure to replace {ACCOUNT_ID}, {REGION}, {REPO} and {IMAGE_TAG} with actual values
DOCKER_IMAGE_ALIASES = {
    "mlflow-image": "{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPO}:{IMAGE_TAG}",
}
