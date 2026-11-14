# RAG Multi Modal Agentique

Activité Pratique N°3 — Pr. Mohamed YOUSSFI

Application RAG (*Retrieval-Augmented Generation*) **multimodale** et **agentique** qui extrait, indexe et interroge le **texte** et les **images** de l'article *Attention Is All You Need* (Vaswani et al., 2017).

Le notebook `rag_multimodal_agentique.ipynb` reprend le code et les explications du professeur sans les modifier. La seule différence concerne les appels au LLM : ils passent par **Ollama en local** au lieu de l'API OpenAI, donc sans clé API payante.

## Technologies

| Rôle | Outil |
|------|-------|
| Gestion de l'environnement | [uv](https://docs.astral.sh/uv/), Python 3.11 |
| Extraction du PDF (texte + images) | PyMuPDF |
| Embeddings texte et image (même espace) | open_clip, CLIP **ViT-B-32** (`laion2b_s34b_b79k`) |
| Découpage du texte | tokeniseur BPE de CLIP, `langchain-text-splitters` |
| Base vectorielle | ChromaDB persistant, distance cosinus |
| LLM local (vision + outils) | Ollama, `qwen3-vl:4b-instruct` (contexte 16 384 tokens) via l'API compatible OpenAI |
| Agent ReAct | LangChain / LangGraph (`create_agent`) |

## Structure du dépôt

```
.
├── rag_multimodal_agentique.ipynb    # notebook principal (exécuté, sorties visibles)
├── pdfs/attention is all you need.pdf
├── architecture_transformer.png      # image de requête (figure 1, page 3 du PDF)
├── Modelfile                         # modèle Ollama qwen3-vl-rag (contexte 16k)
├── pyproject.toml / uv.lock          # dépendances figées
└── README.md
```

Les dossiers `resultats_pdf/` (extraction) et `chroma_pdf/` (index) sont générés par le notebook et ne sont pas versionnés.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/getting-started/installation/) et [Ollama](https://ollama.com/download).

```bash
git clone https://github.com/saaddiri/Activit-Pratique-N-3---RAG-Multi-Modal-Agentique.git
cd Activit-Pratique-N-3---RAG-Multi-Modal-Agentique
uv sync
ollama pull qwen3-vl:4b-instruct
ollama create qwen3-vl-rag -f Modelfile
```

Ollama doit être lancé (application ou `ollama serve`) avant d'exécuter la partie génération.

## Pourquoi Ollama et ce Modelfile ?

- Ollama expose une **API compatible OpenAI** sur `http://localhost:11434/v1`, y compris l'**API Responses** avec images en data URL. Nous l'avons vérifié avant de modifier le code. Le code de génération du professeur (`generer_reponse`) reste donc identique : seuls le client (`base_url`) et le nom du modèle changent, dans la cellule d'initialisation.
- `qwen3-vl:4b-instruct` lit les images (figures de l'article) **et** sait appeler des outils, ce dont l'agent a besoin. La variante *instruct* n'a pas de mode « thinking », un mode trop lent sur CPU.
- Ollama charge les modèles avec un contexte de 4 096 tokens par défaut. Avec jusqu'à 3 figures et une dizaine d'extraits, la requête dépasse cette taille et serait **tronquée silencieusement**. Le `Modelfile` porte le contexte à 16 384 tokens.

Ouvrez ensuite `rag_multimodal_agentique.ipynb` dans VS Code, sélectionnez le noyau `.venv` et exécutez les cellules dans l'ordre. Au premier lancement, les poids CLIP (~600 Mo) sont téléchargés automatiquement.

## Adaptations par rapport au notebook du professeur

Le code et les explications sont identiques au notebook d'origine, à l'exception de **deux cellules de configuration du LLM** :

| Cellule | Original | Adaptation |
|---|---|---|
| Initialisation du modèle génératif | `OpenAI()` + clé API, `gpt-4.1-mini` | `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=1800)`, modèle `qwen3-vl-rag` |
| Création de l'agent | `ChatOpenAI(model=..., temperature=0)` | mêmes paramètres + `base_url`, `api_key` et `timeout` vers Ollama |

Ces adaptations ont été validées par des tests préalables :
- l'API Responses d'Ollama accepte `instructions`, `input_text`, `input_image` (data URL) et `max_output_tokens` ;
- `ChatOpenAI` pointé sur Ollama gère l'appel d'outils (`tool_calls`) et les messages contenant une image (`image_url`) ;
- le délai `timeout=1800` est nécessaire sur CPU : une réponse avec plusieurs figures peut dépasser les 600 s par défaut du client.

## Avancement

- [x] **1. Extraction** : texte par page (`resultats_pdf/textes/`) et images du PDF (`resultats_pdf/images/`)
- [x] **2. Embeddings** : CLIP ViT-B-32, texte et image dans le même espace de 512 dimensions, vecteurs normalisés
- [x] **3. Indexation** : ChromaDB persistant, cosinus, `document_id` SHA-256 pour les réindexations, métadonnées (type, page, source)
- [x] **4. Recherche** : similarité cosinus à partir d'un texte ou d'une image, affichage des k meilleurs résultats
- [x] **Génération** : réponse fondée sur les sources avec un LLM vision local (Ollama)
- [x] **5. Outil agentique** : la recherche RAG encapsulée dans un outil LangChain
- [x] **6. Agent ReAct** : raisonnement → action → observation, graphe compilé avec `create_agent`
- [x] **7. Tests** : requêtes texte, image et mixtes, avec analyse des appels d'outil
