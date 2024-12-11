import json
import tiktoken
import os
from  datetime import datetime
import semchunk
class Utils:
    #Add utility to count output tokens and estimate price.
    def __init__(self):
        pass
    @staticmethod
    def strip_directory_name(file_name):
        file_name_split = file_name.split("/")
        return file_name_split[len(file_name_split)-1]
    @staticmethod
    def get_file_name_only(file_name):
        file_name_with_extension = Utils.strip_directory_name(file_name)
        file_name_with_extension_split = file_name_with_extension.split(".")
        file_name_only = file_name_with_extension_split[0]
        return file_name_only
    @staticmethod
    def get_semantic_chunks(text, model, chunk_size=8000):
        encoder = tiktoken.encoding_for_model(model)
        token_counter = lambda text: len(encoder.encode(text))
        chunks = semchunk.chunk(text, chunk_size=chunk_size, token_counter=token_counter)
        return chunks
    @staticmethod
    def read_json_data(file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)
        return data
    def get_file_list(self,directory):
        file_list = []
        for file in os.listdir(directory):
            file_list.append(file)
        return file_list
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    @staticmethod
    def append_postfix(file):
        datetime_string = datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
        return f"{file}_{datetime_string}"
    @staticmethod
    def clean_binary_string(data):
        return data[2:-1].replace('\\n', '').replace('\\"', '"').replace('\\\\', '\\')
    @staticmethod
    def convert_to_json_from_binary_string(data):
        # Remove the leading "b'" and trailing "'"
        data_str = data[2:-1]

        # Replace escape sequences
        data_str_clean = data_str.replace('\\n', '').replace('\\"', '"').replace('\\\\', '\\')

        # Convert the JSON string to a dictionary
        data_dict = json.loads(data_str_clean)
        return data_dict
    @staticmethod
    def get_file_extension(file_name):
        file_name_split = file_name.split(".")
        #No extension
        extension = file_name
        if len(file_name_split) > 1:
            extension = file_name_split[len(file_name_split)-1]
        return extension
        

