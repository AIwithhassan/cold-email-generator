# Chromadb setup
import csv
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.data = self.read_csv_file(self.file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore2')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
   

    def read_csv_file(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            # Skip the header row
            next(csv_reader)
            for row in csv_reader:
                # Separate technical skills (list) and project link (string)
                skills = tuple(row[:-1])  # Exclude the last element (project link)
                project_link = row[-1]  # Get the last element (project link)
                data.append((skills, project_link))  # Create a tuple with skills and link
        return data


    def load_portfolio(self):
        if not self.collection.count():
            #for value_set in data: 
            for skills, portfolio_url in self.data:
                self.collection.add(
                    documents=str(skills),
                    metadatas = {"portfolio_url": portfolio_url},
                    ids = [str(uuid.uuid4())])
    

    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get("metadatas", [])
                
    

    
