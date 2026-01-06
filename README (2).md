# Multi-PDF Conversational Chatbot using LangChain, Groq LLaMA-3, and Hallucination-Safe RAG

## Overview
This repository implements a robust, modular architecture for multi-document question answering using Retrieval-Augmented Generation (RAG) principles. Built on LangChain, integrated with FAISS vector search, and accelerated by Groq’s LLaMA-3 inference API, the system supports natural language interaction with multiple PDF documents while ensuring high factual reliability and source traceability.
Designed for production deployment, LLM grounding research, and AI-powered document analysis, the pipeline applies:

- Semantic chunking and dense retrieval (MiniLM embeddings)
- Custom threshold filtering to suppress hallucinations
- Conversational memory integration
- Per-response similarity scores and document citations
- Structured logging for auditability

The system is cleanly organized, and CLI-driven, making it extensible to web frontends, enterprise knowledge management, or academic datasets.

| <img src="https://github.com/leovidith/Multi-PDF-Conversational-Chatbot/blob/master/Flowchart.png" width="50%" alt="Segmentation Demo"> |
|:--:|
| *Figure: End-to-end architecture for multi-document question answering using LangChain RAG, Groq LLaMA-3, FAISS retrieval, and hallucination-safe filtering. The pipeline includes PDF ingestion, semantic chunking, embedding, score-thresholded retrieval, conversational memory, and structured response logging.* |

## Objectives
- Enable interactive Q&A over multiple PDF documents via a unified conversational interface ✅
- Implement a Retrieval-Augmented Generation (RAG) pipeline with reliable context retrieval and response generation ✅
- Integrate Groq’s LLaMA-3 API with LangChain to ensure high-speed inference and natural dialogue flow ✅
- Ensure hallucination resistance using a custom score-thresholded retriever with a fallback mechanism ✅
- Provide chunk-level source attribution and similarity scoring for every AI response ✅
- Log each session in both JSON and Markdown formats for auditability and reproducibility ✅
- Maintain a clean, modular codebase with support for .env-based secret management and CLI usage ✅
- Deliver a solution that’s production-ready yet research-aligned, suitable for extensions into web apps or academic RAG studies ✅

## Intuition
Modern document Q&A systems often face a critical challenge: reliably retrieving relevant context from unstructured data (like PDFs) and ensuring the LLM generates grounded, source-faithful answers. Without rigorous filtering or guidance, even powerful models like LLaMA-3 can hallucinate, producing confident but false information — especially when fed loosely matched or noisy chunks.

To address this, the system implements a Retrieval-Augmented Generation (RAG) architecture that separates retrieval from generation. By pairing FAISS for dense vector search with Groq’s high-speed LLaMA-3 via OpenAI-compatible APIs, we ensure that only semantically relevant, document-grounded context is passed to the model. This preserves both response quality and latency performance.

However, retrieval itself isn't immune to error — approximate nearest neighbor (ANN) search can still surface low-quality or borderline-relevant chunks. To mitigate this, a custom ThresholdRetriever was introduced: it filters retrieved chunks based on a minimum similarity score (≥ 0.75). Chunks that don’t meet this confidence threshold are excluded, and if none qualify, the system responds with a controlled fallback ("No reliable answer found") instead of forcing the LLM to fabricate one.

This score-based cutoff ensures that the model only reasons over high-confidence, semantically aligned chunks, reducing hallucination risk without sacrificing conversational ability. The choice of LLaMA-3 (via Groq) further aligns with the goal of low-latency, high-accuracy inference across large context windows — making the entire system robust enough for both exploratory research and real-world applications.
