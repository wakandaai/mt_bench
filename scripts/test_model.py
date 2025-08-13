from src.models.factory import ModelFactory

# Create Toucan model
toucan_model = ModelFactory.create_model('toucan_1.2B')

# Translate
texts = ["Clear all items from the recent documents list"]
translations = toucan_model.translate(texts, source_lang='eng', target_lang='swa')
print("Toucan translation:", translations[0])

# # Create NLLB model  
# nllb_model = ModelFactory.create_model('nllb_200_distilled_600M')
# nllb_translations = nllb_model.translate(texts, source_lang='eng', target_lang='swa')
# print("NLLB translation:", nllb_translations[0])

# Get model info
print("Model info:", toucan_model.get_model_info())