.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/transformers.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/transformers
    .. image:: https://readthedocs.org/projects/transformers/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://transformers.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/transformers/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/transformers
    .. image:: https://img.shields.io/pypi/v/transformers.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/transformers/
    .. image:: https://img.shields.io/conda/vn/conda-forge/transformers.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/transformers
    .. image:: https://pepy.tech/badge/transformers/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/transformers
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/transformers

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

============
Transflate (Transformer Translate)
============
This is my replication from scratch of the article Attentino is all you need. https://arxiv.org/abs/1706.03762

Use-case example is German-to-English machine translation.



Data 
----

The model is trained on 30k English-German translation pairs. https://pytorch.org/text/stable/datasets.html#multi30k 

Tokenizer
----
For simplicity, we use **spacy** tokenizer. https://spacy.io/api/tokenizer



Results
-------
.. list-table:: 
   :widths: 25 25 50
   :header-rows: 1

   * - Source text - German
     - Ground truth - English
     - Model Output
   * - Drei Männer auf Pferden während eines Rennens 
     - Three men on horses during a race
     - Three men on horseback during a race
   * - Ein Kind in einem orangen Shirt springt von Heuballen herunter , während andere Kinder zusehen
     - A child in an orange shirt jumps off bales of hay while other children watch
     - A child in an orange shirt is jumping down a dirty street while other children watch him
   * - Zwei Männer in Shorts arbeiten an einem blauen Fahrrad 
     - Two men wearing shorts are working on a blue bike
     - Two men in shorts are working on a blue bicycle 
   * - Kinder einer Schulklasse sprechen miteinander und lernen
     - Kids conversing and learning in class
     - A group of kids are having a conversation and making a conversation 

Architecture
------------
This project replicates the original architecture of Transformer : Encoder-Deconder model with a lot of attentions.

   
.. image:: https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png
    :width: 300


.. architecture:: = {
        'src_vocab_len' : 8315, # GERMAN vocab size
        'tgt_vocab_len' : 6384, # ENGLISH vocab size
        'N' : 6,                # nb of Transformer layers 
        'd_model' : 512,        # Model size aka Embedding size
        'd_ff' : 2048,          # nb of neurons in Linear layer
        'h' : 8,                # nb of attention heads (thus d_head = 512/8 = 64)
        'p_dropout' : 0.1       # dropout probability (for training)
    }


=====
Reproduce the results
=====

To reproduce the results you will need to train the model (~ 10 min on any standard GPU)


.. prompt::

   python train.py



then set your prefered german sentence

.. transflat.py::

   YOUR_GERMAN_SENTENCE = "Der große Junge geht zur Schule und spricht mit Vögeln"


Finally, run the machine translation

.. prompt::

   python transflate.py


You can also find colab notebooks with similar code and simply execute the cells 🤗


