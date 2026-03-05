# Youtube-Sentiment-Analysis
youtube-sentiment-analysis-with-mlop

you have to write the cridintial in aws configure
so in the terminal write aws configure and then write the s3 configurations

## dom't miss to activate devoloper mode
chrome://extensions/
then load unpacked and choose the templates 
now you have the app as an extension


# deploy
first create folder called .github
and folder inside it called workflows and cicd.yaml file inside them
# End-to-End-Wine-Quality-Prediction


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/entbappy/End-to-End-Wine-Quality-Prediction
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
python app.py
```


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

### with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


### Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

### Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    search in aws on ecr and create repo
    - Save the URI: 657083456427.dkr.ecr.eu-north-1.amazonaws.com/mlops_project
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required
    #installing docker
	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    the repo on github > setting > actions > runner > new self hosted runner> choose linux os> then run command one by one(download and Configure) on aws 
when it asks for Enter the name of runner: [press Enter for ip-172-31-47-112] -> write self-hosted

in the same page in github you have 
Runners/ Add new self-hosted runner · Alaawael3/Youtube-Sentiment-Analysis -> click on the runners to see the active run on github 

# 7. Setup github secrets:
    the repo on github > setting > secrets and variables > Actions > new repositry secret

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  657083456427.dkr.ecr.eu-north-1.amazonaws.com/mlops_project

    ECR_REPOSITORY_NAME = mlops_project
