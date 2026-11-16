package ma.emsi.amal.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {

        String cle = System.getenv("GEMINI_KEY");

        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .maxRetries(0)
                .logRequests(true)
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel =
                new BgeSmallEnV15QuantizedEmbeddingModel();

        // Phase 1 : ingestion des deux documents

        EmbeddingStore<TextSegment> storeIA =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        EmbeddingStore<TextSegment> storeReseaux =
                creerEmbeddingStore("reseaux.pdf", embeddingModel);

        // Phase 2 : création des deux ContentRetriever

        ContentRetriever retrieverIA =
                creerContentRetriever(storeIA, embeddingModel);

        ContentRetriever retrieverReseaux =
                creerContentRetriever(storeReseaux, embeddingModel);

        // Description du contenu des sources

        Map<ContentRetriever, String> descriptions = new HashMap<>();

        descriptions.put(
                retrieverIA,
                "Support de cours portant sur l'intelligence artificielle, "
                        + "le RAG et les modèles d'intelligence artificielle."
        );

        descriptions.put(
                retrieverReseaux,
                "Cours sur les réseaux informatiques : architectures réseau, "
                        + "modèle TCP/IP, protocoles IP, TCP, UDP et DNS."
        );

        // Création du QueryRouter

        QueryRouter queryRouter =
                new LanguageModelQueryRouter(modele, descriptions);

        // Création du RetrievalAugmentor

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // Création de l'assistant

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Test du routage

        Scanner scanner = new Scanner(System.in);

        System.out.println("Posez vos questions (tapez 'exit' pour quitter)");

        while (true) {

            System.out.print("\nQuestion : ");
            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                break;
            }

            String reponse = assistant.chat(question);

            System.out.println("Réponse : " + reponse);
        }

        scanner.close();
    }

    // Méthode pour charger un document et créer son EmbeddingStore

    private static EmbeddingStore<TextSegment> creerEmbeddingStore(
            String nomFichier,
            EmbeddingModel embeddingModel) {

        Document document = ClassPathDocumentLoader.loadDocument(
                nomFichier,
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(document);

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        store.addAll(embeddings, segments);

        System.out.println(
                nomFichier + " : " + segments.size() + " segments indexés."
        );

        return store;
    }

    // Méthode pour créer un ContentRetriever

    private static ContentRetriever creerContentRetriever(
            EmbeddingStore<TextSegment> store,
            EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}
