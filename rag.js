import "dotenv/config";
import { readFile } from "node:fs/promises";

import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const question = process.argv.slice(2).join(" ") || "How much can I spend on my remote office chair?";
const googleApiKey = process.env.GOOGLE_API_KEY;

if (!googleApiKey) {
	throw new Error("Set GOOGLE_API_KEY in the environment before running this script.");
}

const knowledge = await readFile(new URL("./knowledge.txt", import.meta.url), "utf8");

const splitter = new RecursiveCharacterTextSplitter({
	chunkSize: 200,
	chunkOverlap: 20,
});

const documents = await splitter.createDocuments([knowledge]);

const embeddings = new GoogleGenerativeAIEmbeddings({
	model: "gemini-embedding-001",
});

const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);
const retriever = vectorStore.asRetriever({ k: 1 });

const model = new ChatGoogleGenerativeAI({
    apiKey: googleApiKey,
	model: "gemini-3.6-flash",
	temperature: 0,
});

const prompt = ChatPromptTemplate.fromTemplate(`
Answer the question using only the policy context below.
If the answer is not in the context, say you do not know.
If a covered item is part of a total stipend, report that total stipend amount.

Policy context:
{context}

Question: {question}
`);

const chain = RunnableSequence.from([
	{
		context: retriever.pipe(
			new RunnableLambda({
				func: (retrievedDocuments) => retrievedDocuments.map(({ pageContent }) => pageContent).join("\n\n"),
			}),
		),
		question: new RunnablePassthrough(),
	},
	prompt,
	model,
	new StringOutputParser(),
]);

const answer = await chain.invoke(question);
console.log(answer);
