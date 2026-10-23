# RAG local pour Vivado et Vitis

Un moteur de recherche documentaire pour la conception FPGA, et un service qui le branche à un
LLM. Le modèle ne devine plus une option Tcl ou un code d'erreur Vivado : il lit l'extrait qui
convient, le cite, et se tait quand la documentation ne répond pas.

Tout tourne sur la machine. Pas de compte, pas de service en ligne, pas de clé d'API. L'index
tient dans un fichier SQLite qu'on peut copier, sauvegarder ou inspecter avec `sqlite3`.

## Le problème que ça règle

Un LLM placé devant une question Vivado produit une réponse plausible. `reset_run synth_1` existe,
`write_timing_constraints` n'existe pas, et rien dans la sortie ne distingue les deux. Les modèles
de 7 à 20 milliards de paramètres, ceux qu'on fait tourner sur une carte graphique de bureau,
inventent des options avec une aisance qui coûte cher : un script échoue au bout de quarante
minutes de synthèse, ou pire, il passe et contraint mal le design.

La recherche documentaire ne rend pas le modèle plus savant. Elle lui donne, à chaque question,
les cinq ou six paragraphes qui parlent du sujet, avec le nom du fichier d'où ils viennent. Le
reste du travail est un contrat écrit dans le prompt système : répondre uniquement depuis ces
extraits, citer, et écrire « le corpus ne le dit pas » quand la réponse n'y est pas.

## Ce que contient le dépôt

| Dossier | Contenu |
|---|---|
| `rag/` | Le moteur : découpage, index SQLite, recherche hybride, service OpenAI-compatible, tests |
| `docs/` | Le corpus principal : flux Vivado, Tcl, simulation, contraintes XDC, timing, Vitis, dépannage |
| `skills/` | Huit fiches de travail courtes (VHDL synthétisable, cocotb, XDC, Vitis embarqué, etc.) |
| `prompts/` | Les prompts système, dont le contrat d'ancrage |
| `templates/` | Squelettes de projet, de testbench cocotb, de scripts Tcl |
| `examples/` | Sept exemples complets, du compteur au mini-GPU Mandelbrot en VHDL |
| `mcp/` | Un serveur MCP qui expose `search_docs` (et les commandes Vivado) à un agent de code |
| `scripts/` | Vérifications : `verify_all.sh` rejoue l'ensemble, `check_tcl.py` et `verify_vivado.py` contrôlent les scripts Tcl |
| `index/index.db` | Un index déjà construit sur ce corpus (115 documents, 474 morceaux) |
| `training/eval/bench.jsonl` | Trente questions de contrôle, pour mesurer la recherche sans LLM |

Le corpus a été écrit à partir de la documentation publique AMD et de manipulations réelles des
outils. Chaque affirmation sur un comportement de Vivado ou de Vitis porte un lien vers
`docs.amd.com` ou une sortie d'exécution citée. Là où quelque chose n'a pas pu être vérifié
(pas de carte branchée, version d'outil absente), le texte le dit.

## Comment ça marche

Trois idées suffisent à comprendre le reste.

**Le découpage suit la structure du document.** Un fichier Markdown est coupé à ses titres, jamais
au milieu d'un bloc de code, et les sections trop courtes sont recollées à la suivante. Chaque
morceau garde son fil d'Ariane (`document > section > sous-section`), qui part dans l'embedding :
le vecteur sait de quoi parle le morceau, pas seulement ce qu'il dit.

**La recherche est double.** Les vecteurs trouvent « comment relancer une synthèse déjà terminée »
même quand le texte parle de `reset_run synth_1`. BM25, l'index plein texte, retrouve l'inverse :
`Synth 8-327`, `XPAR_GPIO_0_DEVICE_ID`, ces identifiants exacts que les vecteurs lissent. Les deux
listes sont fusionnées par rang (Reciprocal Rank Fusion), ce qui évite d'avoir à calibrer des
scores qui ne sont pas comparables.

**L'index refuse de mentir.** Le modèle d'embedding et sa dimension sont inscrits dans le fichier
à la création. Chercher avec un autre modèle transformerait la similarité en bruit sans lever la
moindre erreur ; le magasin refuse la requête et dit lequel a servi. C'est ce qui permet de
partager un index en sachant ce qu'il contient.

## Mise en route

Il faut Python 3.10 ou plus. Une commande installe tout et construit l'index :

```bash
./scripts/install.sh --workspace ~/mon_projet_fpga
```

Le script crée `.venv/`, y installe le moteur (`rag/`) et le serveur MCP (`mcp/`),
télécharge le modèle d'embedding (130 Mo, une seule fois), indexe le corpus, vérifie qu'une
recherche répond et que le serveur MCP démarre. Il écrit enfin `hermes-mcp.local.yaml`, la
configuration d'agent avec les chemins de la machine. On peut le relancer sans risque : le venv
est réutilisé et l'indexation ne recalcule que les fichiers modifiés.

| Option | Effet |
|---|---|
| `--workspace DOSSIER` | projet FPGA confié au serveur MCP (défaut : le dossier personnel) |
| `--rebuild` | supprime l'index avant de le reconstruire, après un changement de modèle par exemple |
| `--serve` | lance aussi le service OpenAI-compatible en tâche de fond |

L'index construit ainsi vit dans `~/.local/share/fpga-rag/index.db`, hors du dépôt : un
`git add -A` ne peut pas le ramasser, et `index/index.db` reste celui livré. Ce dernier a été
construit avec le serveur d'embeddings de LM Studio (`text-embedding-nomic-embed-text-v1.5`) ;
le garde-fou décrit plus haut empêche de l'interroger avec le modèle embarqué, il ne sert donc
qu'en recherche par mots exacts tant qu'on ne pointe pas `FPGA_RAG_DB` dessus avec LM Studio.

Sur une machine sans carte graphique, le calcul des vecteurs prend deux à trois minutes. Le
modèle tourne sur le CPU, dans le processus, via ONNX, et calcule quatre morceaux à la fois.
Si l'indexation est tuée (`Processus arrêté`, ou `Killed`), c'est la mémoire : l'attention
coûte le carré de la longueur des morceaux, multiplié par la taille du lot. On relance avec un
lot plus petit :

```bash
FPGA_RAG_EMBED_BATCH=1 FPGA_RAG_EMBED_THREADS=2 ./scripts/install.sh
```

Pour faire les étapes à la main, voir [`rag/README.fr.md`](rag/README.fr.md).

## Brancher un agent

La voie recommandée pour un agent comme Hermes, Claude Desktop, Cline ou VS Code est le
serveur MCP : il expose `search_docs` à côté des outils Vivado, et l'agent l'appelle quand il
en a besoin. Pour Hermes :

1. copier le bloc de `hermes-mcp.local.yaml` sous `mcp_servers:` dans `~/.hermes/config.yaml` ;
2. dans une session, taper `/reload-mcp` ;
3. poser une question précise, « que signifie Synth 8-327 ? » : la réponse doit citer un
   fichier de `docs/`.

Hermes ne transmet pas l'environnement du shell aux serveurs qu'il lance. Les variables
`FPGA_RAG_*` doivent donc figurer dans le bloc `env` de la configuration, et c'est ce que le
script y écrit. Sans elles, `search_docs` cherche un autre index, ou refuse la requête si le
modèle d'embedding diffère.

Cette voie fonctionne avec n'importe quel modèle de génération, local ou distant, **à
condition qu'il sache appeler des outils**. La plupart des modèles récents le font. Pour un
modèle qui ne le sait pas, ou un petit modèle qui oublie de chercher avant de répondre, le
service `serve` décrit plus bas se charge de la recherche et injecte les extraits avant
d'appeler le modèle : il suffit de déclarer `http://127.0.0.1:8098/v1` comme fournisseur
OpenAI-compatible dans le client.

Les autres clients ont leur exemple dans [`mcp/configs/`](mcp/configs/).

## Interroger l'index

En ligne de commande, sans aucun LLM :

```bash
python -m fpga_rag search "contraindre une horloge de 100 MHz" --mode hybrid
python -m fpga_rag search "Synth 8-327" --mode keyword
python -m fpga_rag search "innovus" --mode keyword     # pour voir à quoi ressemble un échec
```

`search` affiche les extraits avec leur source et, en bas, les scores. `--json` donne la même
chose en machine.

Comme modèle OpenAI-compatible, pour n'importe quel client de chat :

```bash
python -m fpga_rag serve                 # http://127.0.0.1:8098/v1
```

Trois noms de modèles sont servis :

| Modèle | Ce qu'il fait |
|---|---|
| `fpga-rag` | recherche hybride sur l'index, réponse ancrée et citée |
| `fpga-rag-keyword` | BM25 seul, utile pour un identifiant dont l'orthographe est connue |
| `fpga-rag-direct` | le modèle seul, sans corpus : pour comparer, pas pour travailler |

Le service lit `FPGA_RAG_CHAT_MODEL` (le modèle de génération à appeler) et `FPGA_RAG_CHAT_URL`
(l'URL OpenAI-compatible de ce modèle, par exemple `http://localhost:1234/v1/chat/completions`
pour LM Studio ou Ollama). Il expose aussi `POST /v1/rag/search`, qui rend les extraits seuls,
sans LLM : c'est la façon la plus rapide de voir ce que la base contient vraiment.

Deux détails de conception valent d'être connus. Le service impose son prompt système et ignore
celui du client : deux prompts contradictoires dans une même requête, c'est exactement ainsi que
l'ancrage se perd. Et quand la réponse est « le corpus ne le dit pas », il n'affiche aucune
source : une liste de fichiers sous un refus laisse croire que l'un d'eux a répondu.

## Ce que la recherche vaut, chiffres en main

`training/eval/bench.jsonl` contient trente questions et, pour chacune, les chaînes exactes
qu'une bonne réponse doit contenir. `python -m fpga_rag eval` compte combien de ces chaînes
apparaissent dans les extraits retrouvés. C'est le plafond de ce qu'un modèle pourra citer, et
ça se mesure sans modèle de langage.

Sur ce corpus, avec le modèle d'embedding de LM Studio, k = 6 :

| Mode | Couverture |
|---|---|
| BM25 seul | 111/117, 94,9 % |
| Hybride | 109/117, 93,2 % |
| Hybride, 2 morceaux maximum par fichier | 111/117, 94,9 % |
| Vecteurs seuls | 94/117, 80,3 % |

Le plafond de deux morceaux par fichier est le réglage par défaut, et il n'est pas décoratif :
sans lui, six morceaux d'un même gros fichier occupent tous les créneaux et la réponse ignore le
reste du corpus. Avec lui, toutes les familles de questions atteignent 100 % sauf une
(`agent_workflow`), où deux attentes du banc portent sur des tableaux que le découpage sépare.

Une chose a été essayée puis abandonnée : retirer les mots vides des requêtes BM25. Mesuré sur
les vingt questions françaises du banc, 94,8 % dans les deux cas. BM25 pondère déjà les mots
fréquents par leur rareté, la liste de mots vides n'ajoutait que du code.

## Ce que ce dépôt ne fait pas

Les manuels AMD ne sont pas redistribués ici : ils leur appartiennent, et un dépôt public n'est
pas l'endroit. Les commandes pour les télécharger sur votre machine depuis `docs.amd.com` sont
dans `rag/README.fr.md` (`amd catalog`, `amd select`, `amd fetch`, `amd index`) ; elles ajoutent
des milliers de pages à l'index et changent vraiment la donne sur les questions pointues.

Il n'y a pas de seuil de pertinence. Sur ce corpus, aucun seuil de similarité ne sépare
proprement le pertinent du hors-sujet : les plages se recouvrent. Le refus est donc confié au
contrat dans le prompt, pas à un score. Un reranker externe améliorerait sans doute les cas
difficiles, mais il ajoute une dépendance et n'a pas été mesuré ici.

Enfin, l'index livré ne couvre que ce corpus. Il ne sait rien de votre projet, de vos cartes ni
de vos contraintes maison. C'est fait pour : `python -m fpga_rag add ~/docs/datasheets` indexe
vos propres documents, et ils restent sur votre machine.

## Licence

MIT, voir `LICENSE`. Le corpus peut être réutilisé, modifié et redistribué dans les mêmes
conditions. Les manuels AMD téléchargés par `amd fetch` restent soumis à leurs propres termes et
ne doivent pas être republiés.
