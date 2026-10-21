# fpga_rag : RAG local sur la documentation FPGA

*English: [README.md](README.md)*

Recherche augmentée (RAG) pour ce dépôt : le modèle n'a plus à *savoir* une
option Tcl ou un code d'erreur, il *lit* le bon extrait et le cite. Tout tourne
sur la machine : aucun service en ligne, aucun compte, et l'index est un seul
fichier rangé hors du dépôt.

```
  documents ──► morceaux ──► vecteurs + index plein texte ──► fichier SQLite (index.db)
  (dépôt,        (coupés aux    (modèle d'embedding            │
   manuels AMD,   titres, aux    local)                        │
   tes fichiers)  pages)                                       ▼
                                question ──► recherche hybride ──► k extraits + source
                                             (sens + mots exacts)        │
                                                                         ▼
                                       le LLM répond UNIQUEMENT depuis ces extraits, cite [n]
```

| Élément | Choix | Pourquoi |
|---|---|---|
| Base | **SQLite**, un fichier : texte, **FTS5** (BM25) et **`sqlite-vec`** (vecteurs, cosinus) | rien à lancer, copiable, lisible avec `sqlite3` |
| Recherche | **hybride** : vecteurs + BM25, fusionnés par rang (RRF) | les vecteurs captent le sens, BM25 retrouve `Synth 8-327` ou `XPAR_GPIO_0_DEVICE_ID` |
| Embeddings | `fastembed` dans le processus (par défaut `nomic-embed-text-v1.5-Q`, 130 Mo, CPU) ou tout serveur compatible OpenAI (LM Studio, Ollama, vLLM) | pas besoin de GPU ; on réutilise un serveur GPU s'il existe |
| Mises à jour | empreinte sha256 par document | réindexer ne revectorise que ce qui a changé |
| Garde fou | le modèle d'embedding est inscrit dans l'index | chercher avec un autre modèle renverrait du bruit sans erreur : c'est refusé |

## Installation

Depuis la racine du dépôt (Python 3.10 ou plus, Linux comme le reste du dépôt) :

```bash
python3 -m venv ~/.venv-fpga && . ~/.venv-fpga/bin/activate
pip install -e "rag[all]"        # fastembed + collecte AMD + lecture PDF
```

La première indexation télécharge le modèle d'embedding depuis Hugging Face
(130 Mo, une seule fois, gardé en cache). Sans accès à Hugging Face, passer par
un serveur local (voir *Modèle d'embedding*).

## Démarrage rapide

```bash
python -m fpga_rag index-repo                      # le dépôt : docs, skills, prompts, templates, examples
python -m fpga_rag search "comment relancer une synthèse déjà terminée ?"
python -m fpga_rag status
```

`index-repo` laisse volontairement `training/` de côté : il contient le banc
d'évaluation et ses réponses.

## Ajouter ses propres documents

```bash
python -m fpga_rag add ~/docs/datasheets                 # md, txt, rst, pdf, html, vhd, v, sv, tcl, xdc, py, c, h...
python -m fpga_rag add ~/notes/carte.md --name carte
python -m fpga_rag search "broches d'horloge de référence" --source local:datasheets
python -m fpga_rag remove local:carte
```

**Où vont les données.** Tout (index, cache du modèle, manuels téléchargés) va
dans `~/.local/share/fpga-rag/` (`$XDG_DATA_HOME/fpga-rag`), jamais dans le
dépôt. Changer d'emplacement avec `FPGA_RAG_HOME`. Les documents ajoutés restent
sur la machine : seul leur texte est gardé dans `index.db`, et rien n'est envoyé
ailleurs, sauf si l'on choisit le moteur `openai` pointé vers un serveur
distant. Ne jamais versionner ni partager un `index.db` construit à partir de
documents non publics.

## Documentation AMD (publique, téléchargée en local)

Les manuels appartiennent à AMD et ne sont pas redistribués ici, pas même dans
un dépôt privé : ces commandes reconstruisent le même corpus sur chaque
machine, depuis les points d'accès publics de docs.amd.com (autorisés par son
robots.txt, sans compte).

```bash
python -m fpga_rag amd catalog                 # plus de 3 000 manuels, 16 Mo, une fois
python -m fpga_rag amd select                  # par défaut : guides du flot Vivado/Vitis
python -m fpga_rag amd measure                 # À LIRE : nombre de sujets avant de télécharger
python -m fpga_rag amd fetch                   # ~2 requêtes/s, reprise possible (marqueur .done)
python -m fpga_rag amd index
```

Ou coller directement les liens du navigateur, les deux types sont gérés :

```bash
python -m fpga_rag amd add https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Synthesis
python -m fpga_rag amd add https://docs.amd.com/v/u/en-US/<nom_du_document>_<code>
```

* un lien `/r/...` ajoute le manuel entier à la sélection, le télécharge sujet
  par sujet et l'indexe (la section visée par le lien n'a pas d'importance) ;
* un lien `/v/u/...` désigne un document disponible seulement en PDF
  (spécifications de protocole, anciens guides) : le PDF derrière la page est
  téléchargé puis indexé page par page. Donner `--name CODE` si le lien ne
  contient pas de code.

La sélection par défaut ne dépend d'aucun composant (UG835 référence Tcl,
UG894, UG892, UG900, UG901, UG903, UG904, UG906, UG908, UG949, UG896, UG994,
UG1118, UG1400). Ajouter les manuels de son propre matériel avec
`amd select --codes UG901 PG...`. `measure` signale les manuels de plus de 900
sujets : une référence de registres (des milliers de tableaux de champs de
bits) noie la recherche, mieux vaut l'écarter.

## Modèle d'embedding

| Réglage | Usage |
|---|---|
| par défaut (`fastembed`, `nomic-ai/nomic-embed-text-v1.5-Q`) | CPU, sans serveur. Modèle anglais : bien pour la doc AMD et le code, moins bon sur les tournures françaises |
| `FPGA_RAG_BACKEND=openai`, `FPGA_RAG_BASE_URL=http://localhost:1234`, `FPGA_RAG_MODEL=text-embedding-nomic-embed-text-v1.5` | un modèle d'embedding déjà chargé dans LM Studio (ou Ollama : `http://localhost:11434`) |
| `FPGA_RAG_MODEL=google/embeddinggemma-300m` ou `intfloat/multilingual-e5-large` | multilingue (questions en français), plus lourd (1,2 à 2,2 Go) |
| `FPGA_RAG_BACKEND=hash` | aucun modèle : purement lexical, pour les tests ou une machine sans rien |

Les préfixes de tâche (`search_query:` / `search_document:` pour nomic,
`query:` / `passage:` pour e5) sont appliqués automatiquement. **Changer de
modèle impose de réindexer** : `python -m fpga_rag reset --yes`, puis indexer à
nouveau. L'index refuse une requête faite avec un autre modèle plutôt que de
renvoyer du bruit.

## L'utiliser depuis un LLM

* **Agent avec MCP** (Claude, VS Code, Cline, Hermes...) : le serveur de `mcp/`
  expose `search_docs` et `docs_index_status`. Installer les deux paquets dans
  le même environnement ; le serveur lit les mêmes variables `FPGA_RAG_*`.
* **Client de chat (Open WebUI, LiteLLM, Bionic, n'importe quel client
  OpenAI-compatible)** : `python -m fpga_rag serve` expose l'index comme un
  modèle — c'est la voie recommandée, voir la section suivante.
* **Tout LLM sans outils** : coller la sortie de `search` dans le prompt, avec
  `prompts/quickref.fr.md` comme prompt système (son contrat impose de répondre
  uniquement depuis les extraits).
* **Python** :

  ```python
  from fpga_rag import Searcher, format_hits
  hits = Searcher().search("Common 17-69", k=6)           # aussi : sources=["amd"], mode="keyword"
  contexte = format_hits(hits)                            # extraits + citations, prêts pour un prompt
  ```

## Servir l'index comme un modèle (`serve`)

Un index qui ne vit que dans un terminal ne sert qu'à un endroit. `serve` en fait
un modèle OpenAI-compatible : le client choisit `fpga-rag`, le service lit la
base, injecte les extraits et impose le contrat d'ancrage. La réponse cite
`[n]` et finit par un bloc `Sources :` ; quand la réponse n'est pas dans le
corpus, le modèle écrit `le corpus ne le dit pas` et **aucune source n'est
listée**.

```bash
python -m fpga_rag serve                 # 127.0.0.1:8098
python -m fpga_rag serve --port 8100
```

Sur Windows, les mêmes commandes fonctionnent avec `py -3.11` à la place de
`python3` ; le service écoute toujours sur la boucle locale uniquement.

| Modèle servi | Ce qu'il fait |
|---|---|
| `fpga-rag` | recherche hybride (vecteurs + BM25) sur l'index |
| `fpga-rag-keyword` | BM25 seul, pour un identifiant exact (`Synth 8-327`) |
| `fpga-rag-direct` | le modèle seul, sans corpus — pour comparer, jamais pour répondre |

| Variable | Défaut | Rôle |
|---|---|---|
| `FPGA_RAG_CHAT_MODEL` | `gpt-oss-20b` | modèle de génération côté serveur amont |
| `FPGA_RAG_CHAT_URL` | `FPGA_RAG_BASE_URL` + `/v1/chat/completions` | où ce modèle est servi |
| `FPGA_RAG_PORT` | `8098` | port d'écoute (127.0.0.1 seulement) |
| `FPGA_RAG_K` | `8` | extraits injectés par réponse |
| `FPGA_RAG_MAX_PER_DOC` | `2` | extraits max venant du même fichier |
| `FPGA_RAG_REASONING_EFFORT` | `none` | `none` supprime la réflexion à la source ; `low`/`high` la gardent (transmise dans `reasoning_content`, jamais dans `content`) |

Autres points de l'API : `POST /v1/rag/search` (les extraits seuls, sans LLM —
pratique pour vérifier ce que la base contient), `GET /health` (chemin de
l'index, nombre de morceaux, modèle d'embedding).

### Brancher Open WebUI

1. Paramètres → Connexions → OpenAI, ajouter `http://host.docker.internal:8098/v1`
   (clé quelconque : le service n'authentifie pas, il n'écoute que sur la
   boucle locale). Trois modèles apparaissent alors, préfixés par la connexion.
2. Pour garder un nom unique, créer un modèle workspace dont `base_model_id` est
   `fpga-rag.fpga-rag`. Son prompt système devient inutile : le service impose
   le sien (deux prompts contradictoires, c'est ainsi que l'ancrage se perd).
3. Ne pas attacher de base de connaissances à ce modèle : la recherche est déjà
   faite dans l'index, une seconde couche ne ferait que dupliquer les extraits.

Mesuré sur cette machine (gpt-oss-20b, 32768 jetons, 117 documents / 489
morceaux) : **7,4 s** pour une question couverte en appel direct, **10,1 s** à
travers Open WebUI, réponse contenant le Tcl exact et `[2]`/`[5]` cités. Sur une
question hors corpus (prix d'une licence), la réponse est exactement
« le corpus ne le dit pas », sans chiffre et sans liste de sources.

## Mesurer avant de croire

```bash
python -m fpga_rag eval --mode hybrid      # aussi : --mode keyword, --mode vector, -k 10, -v
```

`eval` passe les 30 questions de `training/eval/bench.jsonl` et compte combien
des chaînes exigées apparaissent dans les extraits retrouvés : le plafond de ce
qu'un modèle peut citer, mesuré sans aucun LLM. La commande refuse de tourner
si `training/` a été indexé (le score ne vaudrait rien).

Mesuré pendant la construction de ce paquet, **avec le seul moteur `hash`, sans
modèle** (dépôt indexé, k = 6) :

| mode | couverture |
|---|---|
| keyword (BM25) | 111/117, 94,9 % |
| hybrid | 97/117, 82,9 % |
| vector (hash, pas sémantique) | 77/117, 65,8 % |

Ces chiffres disent que la chaîne fonctionne et que BM25 seul retrouve déjà
presque toutes les chaînes exactes de ce banc. Ils ne disent rien du vrai
modèle d'embedding : relancer les trois modes avec lui sur ta machine et garder
le meilleur pour ton usage (`--mode` en ligne de commande, `mode=` en Python).
Le banc est fait d'identifiants exacts, ce qui avantage BM25 ; des questions
formulées avec ses propres mots avantagent les vecteurs.

Mesuré depuis, **avec le vrai modèle d'embedding**
(`text-embedding-nomic-embed-text-v1.5` servi par LM Studio, 768 dimensions,
dépôt indexé : 117 documents, 489 morceaux, k = 6) :

| mode | couverture |
|---|---|
| keyword (BM25) | 111/117, 94,9 % |
| hybrid | 109/117, 93,2 % |
| hybrid, 2 morceaux max par fichier | **111/117, 94,9 %** |
| vector | 94/117, 80,3 % |

Deux enseignements, tous deux vérifiés plutôt que supposés :

* **la diversité paie** : sans plafond, six morceaux d'un même fichier peuvent
  remplir tous les créneaux ; à deux par fichier, la couverture passe de 93,2 à
  94,9 % et *toutes* les familles de tâches sauf `agent_workflow` atteignent
  100 %. C'est le défaut (`--par-doc 2`, `--par-doc 0` pour l'ancien
  comportement) ;
* **retirer les mots vides de la requête BM25 n'apporte rien ici** : mesuré sur
  les 20 questions françaises du banc, 94,8 % dans les deux cas. BM25 pondère
  déjà les mots fréquents par leur rareté ; le code reste donc sans liste de
  mots vides, plutôt que d'en ajouter une qui ne sert à rien.

Les deux échecs restants de `agent_workflow` viennent d'attentes du banc
(`PROGRESS=100%`, `Unverified:`) réparties sur des tableaux que le découpage
sépare : c'est une limite du banc, pas de la recherche.

## Non vérifié ici

Écrit dans un environnement sans accès à Hugging Face ni à docs.amd.com :

* le modèle par défaut (`nomic-embed-text-v1.5-Q`) n'a été ni téléchargé ni exécuté ;
* la collecte AMD est testée contre des réponses enregistrées, pas contre le site réel ;
* les codes de manuels par défaut n'ont pas été vérifiés dans le catalogue réel
  (`amd select` affiche chaque manuel trouvé et les codes introuvables).

Les tests (`cd rag && python -m pytest`) couvrent le découpage, l'index SQLite
(mises à jour incrémentales, aucune ligne orpheline, garde fou du modèle), la
recherche hybride, le moteur `openai` face à un serveur HTTP local, la lecture
PDF, la collecte AMD et la ligne de commande.
