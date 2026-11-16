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
import dev.langchain4j.model.input.PromptTemplate;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.util.List;
import java.util.Collection;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test4 {

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

        // Récupération des clés API
        String cleGemini = System.getenv("GEMINI_KEY");
        String cleTavily = System.getenv("TAVILY_API_KEY");

        if (cleGemini == null || cleGemini.isBlank()) {
            throw new RuntimeException("La variable GEMINI_KEY est introuvable.");
        }

        if (cleTavily == null || cleTavily.isBlank()) {
            throw new RuntimeException("La variable TAVILY_API_KEY est introuvable.");
        }

        // Création du modèle de langage
        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cleGemini)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .maxRetries(0)
                .logRequests(true)
                .logResponses(true)
                .build();

        // PHASE 1 : Enregistrement des embeddings du PDF

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

        // 4. Calcul des embeddings
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

        // PHASE 2 : Récupération des informations depuis le PDF et le Web

        // 1. Création du ContentRetriever pour le PDF
        EmbeddingStoreContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 2. Création du moteur de recherche Web Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(cleTavily)
                .build();

        // 3. Création du ContentRetriever pour le Web
        WebSearchContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        // 4. Création du QueryRouter avec les deux retrievers

        // Ancienne version du Test 5 :
        // Le DefaultQueryRouter utilise les deux retrievers
        // pour toutes les questions.
        //
        // DefaultQueryRouter queryRouter =
        //         new DefaultQueryRouter(pdfRetriever, webRetriever);

        // Nouvelle version : QueryRouter personnalisé

        PromptTemplate template = PromptTemplate.from(
                "Est-ce que la question '{{question}}' porte sur les notions " +
                        "expliquées dans le support de cours PDF (RAG, embeddings, " +
                        "fine-tuning et modèles d'intelligence artificielle) ? " +
                        "Si la question concerne une actualité récente ou une information " +
                        "extérieure au contenu du cours, réponds non. " +
                        "Réponds seulement par oui ou non."
        );

        QueryRouter queryRouter = new QueryRouter() {

            @Override
            public Collection<ContentRetriever> route(Query query) {

                String prompt = template.apply(
                        Map.of("question", query.text())
                ).text();

                String decision = modele.chat(prompt).trim();

                System.out.println("Question analysée : " + query.text());
                System.out.println("Décision du LM : " + decision);

                if (decision.equalsIgnoreCase("oui")) {
                    System.out.println("Utilisation du PDF et du Web.");
                    return List.of(pdfRetriever, webRetriever);
                }

                System.out.println("Question extérieure au cours : Web uniquement.");
                return List.of(webRetriever);
            }
        };

        // 5. Création du RetrievalAugmentor
        DefaultRetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // 6. Création de la mémoire pour 10 messages
        MessageWindowChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // 7. Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // 8. Conversation avec l'assistant
        System.out.println("\n=== RAG AVEC PDF ET RECHERCHE WEB ===");
        System.out.println("Posez vos questions sur le cours ou sur le Web.");
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
