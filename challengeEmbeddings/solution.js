import { openai, supabase } from './config.js';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

/*
  Challenge: Text Splitters, Embeddings, and Vector Databases!
    1. Use LangChain to split the content in movies.txt into smaller chunks.
    2. Use OpenAI's Embedding model to create an embedding for each chunk.
    3. Insert all text chunks and their corresponding embedding
       into a Supabase database table.
 */

/* Split movies.txt into text chunks.
Return LangChain's "output" – the array of Document objects. */
async function splitDocument(document) {
  const response = await fetch(document);
  const text = await response.text();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 35,
  });
  const output = await splitter.createDocuments([text]);
  return output;
}

/* Create an embedding from each text chunk.
Store all embeddings and corresponding text in Supabase. */
async function createAndStoreEmbeddings() {
  const chunkData = await splitDocument("movies.txt");
  const data = await Promise.all(
    chunkData.map(async (chunk) => {
      const embeddingResponse = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: chunk.pageContent
      });
      return { 
        content: chunk.pageContent, 
        embedding: embeddingResponse.data[0].embedding 
      }
    })
  );
  await supabase.from('movies').insert(data);
  console.log('SUCCESS!');
}
createAndStoreEmbeddings();