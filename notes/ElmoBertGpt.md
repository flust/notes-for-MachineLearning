## ELMO BERT GPT

word

1-of-N encoding

Word Embedding

一个词会有多种意思

#### 1. ELMO

Embedding from Language Model

RNN-based language models

每一层都会有 context embedding

两个 RNN 得到的 embedding 做 weighted sum 再做 downstream tasks



#### 2. BERT

Bidirectional Encoder Representations from Transformers

BERT = Encoder of Transformer

Learned from a large amount of text without annotation



Training of BERT

Approach1: Masked LM

把输入的句子随机 mask ，让 BERT 去猜测 mask 的词是什么，通过一个 Linear Multi-class Classifier 预测，

Approach2: Next Sentence Prediction

给两个句子，判断是否是被接在一起的，[seq]表示句子之间的符号，[CLS]表示输出结果的位置，做分类，做一个 Linear Binary Classifier，输出 yes / no，



How to use BERT - Case 1

input: single sentence

output: class

Example: Sentiment analysis, Document Classification



How to use BERT - Case 2

input: single sentence

output: class of each word

Example: Slot filling(RNN里讲过)

BERT会把每个词都输出一个embedding



How to use BERT - Case 3

input: two sentences

output: class

Example: Natural Language Inference



How to use BERT - Case 4

Extraction-based Question Answering

给一篇文章，问问题，希望得到答案

输入 D Q 输出 s e

[cls] question [seq] document



#### 3. ERNIE

Enhanced Representation through Knowledge Integration

特别为了中文来设计的

BERT用的中文的字为单位

ERNIE中文以词为单位



Multilingual BERT

Trained on 104 languages

给英文的文章分类，可以自己学会中文文章分类



#### 4. GPT

Generative Pre-Training

非常大

ELMO 94M 个参数

BERT 340M 个参数

GPT-2 1524M 个参数

GPT是Transformer的 Decoder

GPT-2 完全可以直接做 Reading Comprehension / Summarization / Translation