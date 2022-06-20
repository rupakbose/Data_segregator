gcloud builds submit --tag gcr.io/datasetgenerator/datagen  --project=datasetgenerator

gcloud run deploy --image gcr.io/datasetgenerator/datagen --platform managed  --project=datasetgenerator --allow-unauthenticated

