# 一、Attention

首先从Seq2Seq起了解Attention与NLP

### 1.Seq2Seq

seq2seq模型是由编码器（Encoder）和解码器（Decoder）组成的。其中，编码器会处理输入序列中的每个元素，把这些信息转换为一个向量（称为上下文（context））。当我们处理完整个输入序列后，编码器把上下文（context）发送给解码器，解码器开始逐项生成输出序列中的元素。

![encoder-decoder](file://C:\Users\10369\OneDrive\%E6%A1%8C%E9%9D%A2\learn-nlp-with-transformers-main\docs\%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86\pictures\1-3-mt.gif?lastModify=1629306762)

seq2seq模型的编码解码机制一般由循环神经网络RNN来完成，它常用于机器翻译，智能问答等任务，为了解决seq2seq模型在编解码过程中如何得到合适的上下文的问题，引出了attention机制，他模仿人的注意力机制，给与重点部分更大的权重与关注。



### 2.Attention机制

一个注意力模型不同于经典的序列到序列（seq2seq）模型，主要体现在 2 个方面：

首先，编码器会把更多的数据传递给解码器。编码器把所有时间步的 hidden state（隐藏层状态）传递给解码器，而不是只传递最后一个 hidden state（隐藏层状态）:

![img](file://C:\Users\10369\OneDrive\%E6%A1%8C%E9%9D%A2\learn-nlp-with-transformers-main\docs\%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86\pictures\1-6-mt-1.gif?lastModify=1629307150)

其次，第二，注意力模型的解码器在产生输出之前，做了一个额外的处理。为了把注意力集中在与该时间步相关的输入部分。解码器做了如下的处理：

1. 查看所有接收到的编码器的 hidden state（隐藏层状态）。其中，编码器中每个 hidden state（隐藏层状态）都对应到输入句子中一个单词。
2. 给每个 hidden state（隐藏层状态）一个分数。
3. 将每个 hidden state（隐藏层状态）乘以经过 softmax 的对应的分数，从而，高分对应的  hidden state（隐藏层状态）会被放大，而低分对应的  hidden state（隐藏层状态）会被缩小。

![](C:\Users\10369\OneDrive\桌面\learn-nlp-with-transformers-main\docs\篇章2-Transformer相关原理\pictures\1-7-attention-dec.gif)

注意力模型的整个过程：

1. 注意力模型的解码器 RNN 的输入包括：一个embedding 向量，和一个初始化好的解码器 hidden state（隐藏层状态）。
2. RNN 处理上述的 2 个输入，产生一个输出和一个新的 hidden state（隐藏层状态 h4 向量），其中输出会被忽略。
3. 注意力的步骤：我们使用编码器的 hidden state（隐藏层状态）和 h4 向量来计算这个时间步的上下文向量（C4）。
4. 我们把 h4 和 C4 拼接起来，得到一个向量。
5. 我们把这个向量输入一个前馈神经网络（这个网络是和整个模型一起训练的）。
6. 前馈神经网络的输出的输出表示这个时间步输出的单词。
7. 在下一个时间步重复这个步骤。



# 二、Transformer

Transformer 使用了 Seq2Seq任务中常用的结构——包括两个部分：Encoder 和 Decoder。一般的结构图，都是像下面这样：

![image-20210819012508717](C:\Users\10369\AppData\Roaming\Typora\typora-user-images\image-20210819012508717.png)

### 1、宏观理解

transformer主要分成编码和解码两个部分：

encoder由多层编码器组成，每层编码器在结构上都是一样的，但不同层编码器的权重参数是不同的。每层编码器里面，主要由以下两部分组成

- Self-Attention Layer
- Feed Forward Neural Network（前馈神经网络，缩写为 FFNN）

输入编码器的文本数据，首先会经过一个 Self Attention 层，这个层处理一个词的时候，不仅会使用这个词本身的信息，也会使用句子中其他词的信息（你可以类比为：当我们翻译一个词的时候，不仅会只关注当前的词，也会关注这个词的上下文的其他词的信息）。本文后面将会详细介绍 Self Attention 的内部结构。

接下来，Self Attention 层的输出会经过前馈神经网络。

同理，解码器也具有这两层，但是这两层中间还插入了一个 Encoder-Decoder Attention 层，这个层能帮助解码器聚焦于输入句子的相关部分（类似于 seq2seq 模型 中的 Attention）

### 2、细节理解

和通常的 NLP 任务一样，我们首先会使用词嵌入算法（embedding algorithm），将每个词转换为一个词向量

#### Encoder(编码器)

编码器（Encoder）接收的输入都是一个向量列表，输出也是大小同样的向量列表，然后接着输入下一个编码器。

第一 个/层 编码器的输入是词向量，*而后面的编码器的输入是上一个编码器的输出*。

#### Self-Attention 整体理解

当模型处理句子中的每个词时，*Self Attentio*n机制使得模型不仅能够关注这个位置的词，而且能够关注句子中其他位置的词，作为辅助线索，进而可以更好地编码当前位置的词。

![一个词和其他词的attention](C:/Users/10369/OneDrive/桌面/learn-nlp-with-transformers-main/docs/篇章2-Transformer相关原理/pictures/2-attention-word.png)
图：一个词和其他词的attention

如上图可视化图所示，当我们在第五层编码器中（编码部分中的最后一层编码器）编码“it”时，有一部分注意力集中在“The animal”上，并且把这两个词的信息融合到了"it"这个单词中。

# 三、总结

根据课程所给的文本图片资料了解了attention和transformer的大概机制与原理，具体细节还有待进一步根据图像进行理解。
