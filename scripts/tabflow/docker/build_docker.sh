#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <repo_name> <tag> <source_account> <target_account> <region>"
    exit 1
fi

REPO_NAME=$1
TAG=$2
SOURCE_ACCOUNT=$3
TARGET_ACCOUNT=$4
REGION=$5

# Login to AWS ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${SOURCE_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${TARGET_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Build the Docker image
docker build --no-cache -t ${REPO_NAME} -f ./Dockerfile_SM ../..

# Tag the Docker image
docker tag ${REPO_NAME}:latest ${TARGET_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}

# Push the Docker image to the repository
docker push ${TARGET_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}