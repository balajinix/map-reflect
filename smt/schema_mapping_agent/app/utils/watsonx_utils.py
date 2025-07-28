from dotenv import load_dotenv
# from genai import Credentials, Client
# from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models import ModelInference
import os

# def get_bam_response(prompt, model_id, max_new_tokens=600, stop_sequences=["\n\n", "<|endoftext|>"]):
#     load_dotenv()
#     credentials = Credentials.from_env()
#     parameters = TextGenerationParameters(decoding_method="greedy", max_new_tokens=max_new_tokens, temperature=0.05, stop_sequences=stop_sequences, include_stop_sequence=False)
#     client = Client(credentials=credentials)
#     responses = list(
#         client.text.generation.create(
#             model_id=model_id,
#             inputs=[prompt],
#             parameters=parameters,
#         )
#     )
#     return responses[0].results[0].generated_text


def get_wxai_response(prompt, model_id, max_new_tokens=1000, stop_sequences=["\n\n", "<|endoftext|>"]):
    load_dotenv()
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=os.environ["WATSONX_APIKEY"]
    )
    project_id = os.environ["WATSONX_PROJECT_ID"]


    parameters = {
        GenParams.MIN_NEW_TOKENS: 0,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.REPETITION_PENALTY: 1,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    model = ModelInference(
        model_id=model_id, 
        params=parameters, 
        credentials=credentials,
        project_id=project_id)
    
    while True:
        try:
            response = model.generate_text(prompt=[prompt])
            return response[0]
        except:
            continue
    # return response[0]