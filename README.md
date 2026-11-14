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
| LLM local (vision + outils) | Ollama, `qwen3-vl:4b-instruct` |
| Agent ReAct | LangChain / LangGraph (`create_agent`) |

## Structure du dépôt

```
.
├── rag_multimodal_agentique.ipynb    # notebook principal (exécuté, sorties visibles)
├── pdfs/attention is all you need.pdf
├── architecture_transformer.png      # image de requête (figure 1, page 3 du PDF)
├── pyproject.toml / uv.lock          # dépendances figées
└── README.md
```

Les dossiers `resultats_pdf/` (extraction) et `chroma_pdf/` (index) sont générés par le notebook et ne sont pas versionnés.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/getting-started/installation/) et [Ollama](https://ollama.com/download).

```bash
git clone <url-du-depot>
cd rag-multimodal-agentique
uv sync
ollama pull qwen3-vl:4b-instruct
```

Ouvrez ensuite `rag_multimodal_agentique.ipynb` dans VS Code, sélectionnez le noyau `.venv` et exécutez les cellules dans l'ordre. Au premier lancement, les poids CLIP (~600 Mo) sont téléchargés automatiquement.

## Avancement

- [x] **1. Extraction** : texte par page (`resultats_pdf/textes/`) et images du PDF (`resultats_pdf/images/`)
- [x] **2. Embeddings** : CLIP ViT-B-32, texte et image dans le même espace de 512 dimensions, vecteurs normalisés
- [x] **3. Indexation** : ChromaDB persistant, cosinus, `document_id` SHA-256 pour les réindexations, métadonnées (type, page, source)
- [ ] **4. Recherche** : similarité cosinus à partir d'un texte ou d'une image, affichage des k meilleurs résultats
- [ ] **Génération** : réponse fondée sur les sources avec un LLM vision local (Ollama)
- [ ] **5. Outil agentique** : la recherche RAG encapsulée dans un outil LangChain
- [ ] **6. Agent ReAct** : raisonnement → action → observation, graphe compilé avec `create_agent`
- [ ] **7. Tests** : requêtes texte, image et mixtes, avec analyse des appels d'outil
