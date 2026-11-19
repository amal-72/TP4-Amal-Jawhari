# Utilisation de l'intelligence artificielle (IA)

Pendant la réalisation du TP4 RAG, j'ai utilisé l'intelligence artificielle ponctuellement pour comprendre le fonctionnement du RAG, analyser les résultats de mes tests et mieux comprendre les mécanismes de transformation et de routage des requêtes.

---

## Utilisation 1 : Comprendre le fonctionnement du RAG

### 1. Problème rencontré

Lors du premier test, j'ai chargé un document PDF contenant un cours sur le RAG. Le document a été découpé en 355 segments, puis transformé en embeddings afin de permettre la recherche des passages pertinents.

Je voulais comprendre le rôle de chaque étape et la manière dont le modèle utilise les informations récupérées pour répondre aux questions.

### 2. Raison de l'utilisation de l'IA

J'ai demandé à l'IA de m'expliquer le fonctionnement du RAG, notamment le chargement des documents, leur découpage, la génération des embeddings et la recherche des passages pertinents.

### 3. Prompt utilisé

« Peux-tu m'expliquer le fonctionnement du RAG dans LangChain4j ? Quel est le rôle du découpage des documents, des embeddings, de la base vectorielle et du ContentRetriever dans la génération d'une réponse ? »

### 4. Réponse de l'IA

Le RAG repose sur plusieurs étapes :

- Le chargement des documents permet de récupérer le contenu à exploiter.
- Le découpage transforme les documents en petits segments appelés chunks.
- Les embeddings transforment les segments en vecteurs numériques.
- La base vectorielle stocke ces vecteurs pour permettre une recherche par similarité.
- Le `ContentRetriever` recherche les segments les plus pertinents par rapport à la question.
- Le modèle de langage utilise les passages récupérés comme contexte pour générer une réponse.

### 5. Ce que j'ai appris

J'ai compris que le RAG permet à un modèle de langage d'utiliser des informations provenant de documents externes. J'ai également compris l'importance du découpage et des embeddings pour retrouver les informations pertinentes.

---

## Utilisation 2 : Diagnostiquer les résultats du RAG naïf

### 1. Problème rencontré

Pendant le premier test, certaines réponses générées à partir du document PDF n'étaient pas suffisamment précises. Par exemple, pour certaines questions sur la définition du RAG ou sur le code LangChain4j, le modèle n'a pas réussi à retrouver une réponse explicite dans les passages fournis.

### 2. Raison de l'utilisation de l'IA

J'ai utilisé l'IA pour analyser les résultats obtenus et comprendre pourquoi le modèle ne répondait pas correctement à certaines questions, alors que le document avait bien été chargé et indexé.

### 3. Prompt utilisé

« Pourquoi un système RAG peut-il ne pas répondre correctement à une question alors que le document PDF a été chargé et que ses embeddings ont été générés ? Quel est le rôle des passages récupérés dans la qualité de la réponse ? »

### 4. Réponse de l'IA

La qualité d'une réponse RAG dépend notamment de la pertinence des passages récupérés. Même si le document est correctement indexé, le retriever peut ne pas retrouver le passage contenant l'information recherchée.

Le modèle peut également ne pas répondre lorsqu'une information n'est pas présente dans le document fourni. Cela permet d'éviter de produire une réponse non justifiée par la source documentaire.

### 5. Ce que j'ai appris

J'ai compris que la génération des embeddings ne garantit pas automatiquement une réponse correcte. Il faut également que les passages récupérés soient pertinents et contiennent suffisamment d'informations pour répondre à la question.

---

## Utilisation 3 : Comprendre le rôle du CompressingQueryTransformer

### 1. Problème rencontré

Lors du test de transformation des requêtes, j'ai posé deux questions successives :

- « Qu'est-ce que le fine-tuning ? »
- « Dans le support de cours fourni, à quoi c'est comparé ? »

La deuxième question était ambiguë, car l'expression « c'est » faisait référence à la première question. Sans transformation, le retriever ne disposait pas forcément de suffisamment de contexte pour comprendre cette référence.

### 2. Raison de l'utilisation de l'IA

J'ai demandé à l'IA d'expliquer pourquoi une question dépendante de l'historique de la conversation peut poser problème lors de la recherche documentaire et comment le `CompressingQueryTransformer` permet de résoudre ce problème.

### 3. Prompt utilisé

« Pourquoi le retriever peut-il avoir des difficultés avec une question comme “Dans le support de cours fourni, à quoi c'est comparé ?” après une première question sur le fine-tuning ? Comment le CompressingQueryTransformer utilise-t-il l'historique de la conversation pour reformuler cette question ? »

### 4. Réponse de l'IA

Le retriever reçoit principalement la question à rechercher. Si celle-ci dépend d'une question précédente, elle peut être difficile à interpréter sans reformulation.

Le `CompressingQueryTransformer` utilise l'historique de la conversation pour transformer la question en une requête autonome.

Dans mon test, la question initiale :

« Dans le support de cours fourni, à quoi c'est comparé ? »

a été reformulée dans les logs en :

« Dans le support de cours fourni, à quoi le fine-tuning est-il comparé ? »

La recherche peut alors retrouver des passages pertinents mentionnant notamment le soft-prompting, T-few, LoRA, les adapters et le continual pre-training.

### 5. Ce que j'ai appris

J'ai compris que le `CompressingQueryTransformer` améliore la recherche documentaire lorsqu'une question dépend du contexte précédent. Il permet de reformuler les questions ambiguës avant de les transmettre au retriever.

---

## Utilisation 4 : Comprendre le routage entre plusieurs sources documentaires

### 1. Problème rencontré

Lors du test de routage, j'ai utilisé deux documents PDF différents :

- `rag.pdf`, contenant 355 segments sur l'intelligence artificielle et le RAG.
- `reseaux.pdf`, contenant 215 segments sur les réseaux informatiques.

Je voulais comprendre comment le système pouvait choisir automatiquement le document adapté à chaque question.

### 2. Raison de l'utilisation de l'IA

J'ai demandé à l'IA d'expliquer le fonctionnement du `LanguageModelQueryRouter` et la manière dont le modèle sélectionne le retriever correspondant à la question.

### 3. Prompt utilisé

« Comment fonctionne le LanguageModelQueryRouter dans LangChain4j lorsqu'on dispose de plusieurs ContentRetriever ? Comment le modèle choisit-il la source documentaire la plus adaptée à une question ? »

### 4. Réponse de l'IA

Le `LanguageModelQueryRouter` utilise un modèle de langage pour analyser la question et choisir le ou les retrievers les plus adaptés parmi les sources disponibles.

Dans mon test, les résultats observés étaient :

- Pour la question « Qu'est-ce que le RAG et comment fonctionne-t-il ? », le modèle a retourné `2`, correspondant au document `rag.pdf`.
- Pour la question « Quelle est la différence entre les protocoles TCP et UDP ? », le modèle a retourné `1`, correspondant au document `reseaux.pdf`.

Le routeur s'appuie sur les descriptions des sources documentaires pour effectuer son choix.

### 5. Ce que j'ai appris

J'ai compris que le routage permet de travailler avec plusieurs sources documentaires sans interroger systématiquement tous les documents. J'ai également constaté que le modèle peut indiquer qu'une information n'est pas présente dans le document sélectionné, plutôt que d'inventer une réponse.

---

## Utilisation 5 : Comprendre quand ne pas utiliser le RAG

### 1. Problème rencontré

Lorsqu'un utilisateur pose une question simple comme « Bonjour », il n'est pas nécessaire de rechercher des informations dans un document PDF.

Je voulais donc comprendre comment éviter d'effectuer une recherche documentaire inutile pour ce type de question.

### 2. Raison de l'utilisation de l'IA

J'ai demandé à l'IA d'expliquer comment mettre en place un routage personnalisé permettant de décider si une question nécessite ou non l'utilisation du RAG.

### 3. Prompt utilisé

« Comment créer un QueryRouter personnalisé dans LangChain4j pour décider si une question nécessite une recherche documentaire ? Comment retourner une liste vide de retrievers lorsqu'une question comme “Bonjour” ne nécessite pas de RAG ? »

### 4. Réponse de l'IA

Il est possible d'utiliser un modèle de langage pour analyser la question et décider si une recherche documentaire est nécessaire.

Si la question nécessite le contenu d'un document, le routeur sélectionne le retriever approprié.

Dans le cas contraire, il retourne une liste vide de retrievers. Le modèle peut alors répondre directement sans utiliser de contexte documentaire.

Dans mon test, la question « Bonjour » a été reconnue comme ne nécessitant pas de RAG. Le programme a affiché « Pas de RAG », puis l'assistant a répondu directement :

« Bonjour ! Comment puis-je vous aider aujourd'hui ? »

### 5. Ce que j'ai appris

J'ai compris qu'un système RAG ne doit pas obligatoirement effectuer une recherche documentaire pour chaque question. Un routage personnalisé permet d'éviter les recherches inutiles et de rendre l'application plus adaptée aux différentes demandes des utilisateurs.

---

## Utilisation 6 : Comprendre le Web RAG et le routage des sources

### 1. Problème rencontré

Dans le dernier test, j'ai combiné deux sources d'informations :

- Un document PDF local contenant le cours sur le RAG.
- Une recherche sur Internet utilisant Tavily.

Le `DefaultQueryRouter` permettait d'utiliser les deux retrievers. Cependant, pour une question concernant les dernières actualités de l'intelligence artificielle, des passages du document PDF pouvaient également être récupérés, même s'ils n'étaient pas directement pertinents.

### 2. Raison de l'utilisation de l'IA

J'ai demandé à l'IA d'expliquer le fonctionnement du Web RAG, le rôle de Tavily et la limite du `DefaultQueryRouter`. Je voulais également comprendre comment sélectionner uniquement la source adaptée à la question.

### 3. Prompt utilisé

« Dans LangChain4j, comment fonctionne le Web RAG avec TavilyWebSearchEngine et WebSearchContentRetriever ? Pourquoi le DefaultQueryRouter peut-il récupérer à la fois des passages PDF et des résultats Web pour une question d'actualité ? Comment utiliser un routeur personnalisé pour sélectionner uniquement la source pertinente ? »

### 4. Réponse de l'IA

Le Web RAG permet de compléter les connaissances d'un modèle avec des informations récupérées sur Internet.

Le `TavilyWebSearchEngine` effectue une recherche Web et le `WebSearchContentRetriever` permet d'intégrer les résultats dans le processus RAG.

Le problème du `DefaultQueryRouter` est qu'il transmet la requête aux différents retrievers configurés. Il peut donc récupérer des passages du PDF même lorsque la question concerne uniquement une actualité récente.

Une solution consiste à utiliser un routeur personnalisé qui sélectionne la source adaptée :

- Pour une question sur le cours, utiliser le retriever PDF.
- Pour une question sur l'actualité, utiliser le retriever Web.
- Pour une question générale ne nécessitant aucune recherche, ne pas utiliser de retriever.

### 5. Ce que j'ai appris

J'ai compris que le Web RAG permet d'utiliser des informations plus récentes que celles contenues dans un document local. J'ai également appris que le choix des sources est essentiel pour éviter d'ajouter des informations inutiles au contexte transmis au modèle.

---

## Conclusion

Ces différentes utilisations m'ont permis de mieux comprendre le fonctionnement du RAG et les mécanismes proposés par LangChain4j pour améliorer la recherche documentaire.

J'ai retenu que la qualité d'une réponse dépend non seulement du modèle de langage, mais également de la pertinence des documents, de la transformation des requêtes et du choix des sources utilisées.

Ce TP m'a également permis de comprendre l'intérêt des routeurs personnalisés pour éviter les recherches inutiles et adapter le fonctionnement du système aux différents types de questions.