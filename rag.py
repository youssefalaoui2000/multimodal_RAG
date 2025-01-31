
import spaces
import gradio as gr
import pdfplumber
import pandas as pd
import uuid
from langchain_chroma import Chroma
import chromadb

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore

#from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn

from PIL import Image
import os



class ChatbotRAG:
    def __init__(self):
        
        self.retriever = None
        self.device_0 = torch.device('cuda:0')  #  pour Llava
        self.device_1 = torch.device('cuda:1')  #  pour le reader LLM
        self.device_2 = torch.device('cuda:2')  #  pour le summarizer
        self.load_models() 
        
        
    def load_models(self):
        
        # modèle multimodal llava

        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16
        )
        
        self.llava_model.to(self.device_0)
      
        
        # pipeline de résumé de texte
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=2) #device 2

        # modèle d'embedding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore = Chroma(
            collection_name="summaries",
            embedding_function=self.embedding_model,
            persist_directory="/data/chroma",
        )
        
        # modèle de langage
        READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",              #Normal Float 4-bit pour le sctockage des poids
            bnb_4bit_compute_dtype=torch.bfloat16,  #bfloat16 pour les calculs
        )
        
        self.reader_model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME,
            #torch_dtype=torch.float16, #redondant si quantification
            quantization_config=bnb_config,
        )#.to(self.device_1)  #Commenté car géré par Accelerate lors de la quantification
        
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        
        self.READER_LLM = pipeline(
            model=self.reader_model,
            tokenizer=self.tokenizer,
            task="text-generation",
            #device=1,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=400,
        )
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """En utilisant les informations contenues dans le contexte,
fournissez une réponse complète à la question.
La réponse est en français si la question est en français. Répondez uniquement à la question posée ; la réponse doit être concise, structurée et pertinente par rapport à la question.""",
            },
            {
                "role": "user",
                "content": """Contexte:
{context}
---
Voici maintenant la question à laquelle vous devez répondre.

Question: {question}""",
            },
        ]
        self.RAG_PROMPT_TEMPLATE = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

        print("Modèles chargés avec succès.")
    
    def clean_table(self, table):
        df = pd.DataFrame(table)
        df_cleaned = df.dropna(axis=1, how='all')
        df_cleaned = df_cleaned.dropna(how='all')
        return df_cleaned

    def extract_content_from_pdf(self, pdf_path):
        text_content = []
        tables_content = []
        images_content = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content.append(page.extract_text())

                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        df_cleaned = self.clean_table(table)
                        tables_content.append(df_cleaned.to_markdown(index=False))

                for image in page.images:
                    x0 = image['x0']
                    top = image['top']
                    x1 = image['x1']
                    bottom = image['bottom']
    
                    # extraire précisément l'image
                    pil_image = page.crop((x0, top, x1, bottom)).to_image(resolution=150).original
                  
                    images_content.append(pil_image)

        return text_content, tables_content, images_content

    def generate_image_caption(self, pil_image):

        #[INST] encadre les instructions
        
        prompts = "[INST] <image>\n Give a detailed technical description of what's is in the image, including the colors of the primary elements.  [/INST]"
        
        inputs = self.processor(
            text=prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device_0)
        
        outputs = self.llava_model.generate(**inputs, max_new_tokens=400)
        
        # sortie du decoder
        descriptions = self.processor.decode(outputs[0], skip_special_tokens=True) 
        return descriptions


    def summarize_text(self, text):
        return self.summarizer(text, max_length=100, min_length=40, do_sample=False)[0]['summary_text']

    def summarize_table(self, table_text):
        return self.summarizer(table_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

    def process_pdf_and_create_retriever(self, pdf_paths):
        all_text_content = []
        all_tables_content = []
        all_images_content = []

        for pdf_path in pdf_paths:
            text_content, tables_content, images_content = self.extract_content_from_pdf(pdf_path)
            all_text_content.extend(text_content)
            all_tables_content.extend(tables_content)
            all_images_content.extend(images_content)

        text_summaries = [self.summarize_text(text) for text in all_text_content]
        #On prend la description tel que le contenu original
        image_texts = [self.generate_image_caption(image) for image in all_images_content]
        #On résume la description de l'image
        image_summaries = [self.summarize_text(text) for text in image_texts]
        table_summaries = [self.summarize_table(table) for table in all_tables_content if table is not None and table != [None]]

        all_raw_content = all_text_content + all_tables_content + image_texts
        all_summaries = text_summaries + table_summaries + image_summaries

        store = InMemoryByteStore()
        id_key = "doc_id"

        #Instanciation du multivector
        retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=store,
            id_key=id_key,
        )

        doc_ids = [str(uuid.uuid4()) for _ in all_raw_content]

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(all_summaries)
        ]
        #Ajout des résumé comme les documents enfants 
        retriever.vectorstore.add_documents(summary_docs)

        raw_doc_objects = [
            Document(page_content=raw_doc, metadata={id_key: doc_ids[i]})
            for i, raw_doc in enumerate(all_raw_content)
        ]
        #Ajout des documents parents
        retriever.docstore.mset(list(zip(doc_ids, raw_doc_objects)))
       
        self.retriever = retriever

    def chunk_text(self, text, max_chunk_size=512):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False)
        input_ids = tokens['input_ids'][0]
        chunked_input_ids = [input_ids[i:i+max_chunk_size] for i in range(0, len(input_ids), max_chunk_size)]
        chunked_texts = [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunked_input_ids]
        return chunked_texts

    def answer_with_rag(
        self,
        question: str,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
        search_type: str = "similarity",
    ):
        if self.retriever is None:
            return "Veuillez d'abord créer le retriever en téléchargeant des PDF."
        relevant_docs = self.retriever.vectorstore.search(
            query=question, k=num_retrieved_docs, search_type=search_type
        )

        final_docs = []
        for doc in relevant_docs:
            
            #Récupération du document parent
            parent_doc_id = doc.metadata["doc_id"]
            parent_doc = self.retriever.docstore.mget([parent_doc_id])[0]
            
            if parent_doc:
                #On découpe les documents pour respecter la taille de 512 tokens
                parent_chunks = self.chunk_text(parent_doc.page_content)
                final_docs.extend(parent_chunks)

        final_docs = final_docs[:num_docs_final]

        context = "\nDocuments extraits:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(final_docs)]
        )

        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)

        answer = self.READER_LLM(final_prompt)[0]["generated_text"]

        return answer
