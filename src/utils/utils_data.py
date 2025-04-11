from src.examples.extraction_examples import get_identifier_examples, get_denomination_examples, get_valorisation_examples, get_marketcap_examples, get_corporateact_examples, get_characteristic_examples
import os
import json

def checking_completitud_examples_in_functions2prompt():
    list_of_functions = [get_identifier_examples, get_denomination_examples, get_valorisation_examples, get_marketcap_examples, get_corporateact_examples, get_characteristic_examples]
    print('--------------------------------------')
    for funct in list_of_functions:
        res  = funct()
        for element_id, element in enumerate(res):
            json_with_feat = element['json']
            percent_ = sum([str(v).strip() == "" for k,v in json_with_feat.items()])/len(json_with_feat)
            print(f"{funct.__name__=}/Example {element_id=}/{percent_=}")
        print('--------------------------------------')

def checking_completitud_in_labelized_examples():
    list_of_path4labelized_examples = os.listdir('./ground-truth')
    list_feat = ['Identifier', "Denomination", "Valorisation",  "MarketCap", "CorporateAction", "Characteristics"]
    for feat in list_feat: 
        for id_ex, file in enumerate(list_of_path4labelized_examples):
            file_path = f'./ground-truth/{file}'
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            if feat in ('Identifier', "CorporateAction"):
                json_output_fields = metadata[feat][0]
            else:
                json_output_fields = metadata[feat]
            percent_ = sum([str(v).strip() == "" for k,v in json_output_fields.items()])/len(json_output_fields)
            print(f"{feat=}/Example {id_ex=}/{percent_=}")
        print('--------------------------------------')
        
if __name__ == "__main__":
    checking_completitud_in_labelized_examples()