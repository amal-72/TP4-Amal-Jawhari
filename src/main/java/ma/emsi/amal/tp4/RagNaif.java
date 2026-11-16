package ma.emsi.amal.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaif {

    // Configuration du logging LangChain4j
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {

        // Activation du logging
        configureLogger();

        // Récupération de la clé API
        String cle = System.getenv("GEMINI_KEY");

        if (cle == null || cle.isBlank()) {
            throw new RuntimeException("La variable GEMINI_KEY est introuvable.");
        }

        // Création du modèle de langage
        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .maxRetries(0)
                .logRequests(true)
                .logResponses(true)
                .build();

        // PHASE 1 : Enregistrement des embeddings

        // 1. Chargement du document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("Document chargé avec succès !");

        // 2. Découpage du document en segments
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30)
                .split(document);

        System.out.println("Nombre de segments créés : " + segments.size());

        // 3. Création du modèle d'embeddings BGE
        EmbeddingModel embeddingModel =
                new BgeSmallEnV15QuantizedEmbeddingModel();

        System.out.println("Modèle d'embeddings BGE créé avec succès !");

        // 4. Calcul des embeddings pour les segments
        Response<List<Embedding>> embeddingsResponse =
                embeddingModel.embedAll(segments);

        List<Embedding> embeddings = embeddingsResponse.content();

        System.out.println("Nombre d'embeddings créés : " + embeddings.size());

        // 5. Création du magasin d'embeddings en mémoire
        InMemoryEmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        // 6. Enregistrement des embeddings et des segments
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings et segments enregistrés dans le magasin !");

        // PHASE 2 : Utilisation des embeddings pour répondre aux questions

        // 1. Création du ContentRetriever
        EmbeddingStoreContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 2. Création de la mémoire pour 10 messages
        MessageWindowChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // 3. Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // 4. Conversation avec l'assistant
        System.out.println("\n=== RAG NAÏF ===");
        System.out.println("Posez vos questions sur le support de cours.");
        System.out.println("Tapez 'exit' pour terminer.\n");

        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {
                System.out.print("Vous : ");
                String question = scanner.nextLine();

                if (question.equalsIgnoreCase("exit")) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);

                System.out.println("\nAssistant : " + reponse + "\n");
            }
        }

        System.out.println("Fin de la conversation.");
    }
}
