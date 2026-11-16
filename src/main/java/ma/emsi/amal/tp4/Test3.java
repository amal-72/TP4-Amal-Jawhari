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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Test3 {

    public static void main(String[] args) {

        String cle = System.getenv("GEMINI_KEY");

        // Création du modèle de langage
        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .maxRetries(0)
                .logRequests(true)
                .logResponses(true)
                .build();

        // Phase 1 : ingestion du document

        EmbeddingModel embeddingModel =
                new BgeSmallEnV15QuantizedEmbeddingModel();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(document);

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        store.addAll(embeddings, segments);

        System.out.println(segments.size() + " segments indexés.");

        // Phase 2 : création du ContentRetriever

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // Création du QueryRouter personnalisé

        PromptTemplate template = PromptTemplate.from(
                "Est-ce que la requête '{{question}}' porte sur l'IA ? " +
                        "Réponds seulement par 'oui', 'non' ou 'peut-être'."
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

                if (decision.equalsIgnoreCase("non")
                        || decision.toLowerCase().startsWith("non ")) {

                    System.out.println("Pas de RAG.");

                    return List.of();
                }

                System.out.println("Utilisation du RAG.");

                return List.of(contentRetriever);
            }
        };

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
}
