unit Notes;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}
{        Input Train        Input Query        Output
 Raw                        QueryString
 Bytes   Corpus             QueryCorpus
 Token   TokenizedCorpus    QueryTokenized     QueryOutput
 Folder layout: WorkRoot \corpus \lists \logs \merges \models \scratch \symbols \tokens }

Good models.
Model gibbon816, at about 10mb. Good as of 818.

Model ts20mb818, down to 2.316 at LR= .000005 on 8-19-26. Using slow AdamW LR schedule. Almost all betters. Stopped, then restarted at 7 pm on 8/19.
On restart, LR started much higher (0.0001) than where it ended (0.0000015). Fixed, so now override continues, but adaptivenest dopes not continue into
next start. Ended with Loss = 1.876 om 8/21. Eventually save adaptive LR value to checkpoint. This is WesTokenize.

Model ts20mb822g. ChatGPT2 with 50,260 vocab. Stared with Loss=2.98. Adaptive LR at 0.0001. Loss going down fast over first 4 epochs to 2.037. This is a ChatGPT2 tokenization.
In epoch 34, down to Loss=1.66. Seconds = 2068. Epoch 50, 1.60. All betters so far.
Would the LR start so low with WesTokenize and 50K symbols?

Modelts20mb824s50 is a WesTokenize model with 50260 symbols. It started with Loss = 8.5!Same Lr of 0.0001, that is, adaptive. At epoch 15, down to Loss =3.3.
Training speed = 4900. Afyer 72 epochs, Loss=2.29.

Model tem77. Using ChatGPT2, and damendthing.txt, I had an initial Loss of >9, then down to 8 in epoch 0, then quickly down to 1.
What dos this mean?

1. For training, add Extrafield1 and ExtraField2, etc. in CheckPoint. Also add saved AdaptiveLR.  Also add CorpusByteCount.

2.  Systematize nTC.
nRawTokenizedCorpus := Length(TokenizedCorpus);
PadToSeqMultiple(TokenizedCorpus, SeqLen);
nTokenizedCorpus := Length(TokenizedCorpus);


3. Replace nSymbols with nVocab.
   Use nTokenizedCorpus instead of Length(TC)

5. Notes on file naming.
  Proc          Input                      Output
  LoadCorpus    CorpusFileName             Corpus
  Symbolize     CorpusFileName             SymbolTable
                SymbolFileName
  Wes Tokenize  SymbolTable                TokenizedCorpus
  GPT Tokenize  SymbolTable                TokenizedCorpus
                MergeTable/MergeFileName
  Embed         TokenizedCorpus            Sequence, Embeddings
  Transform     Sequence                   Embeddings
                Embeddings
  RunForward    UserInput                  Output

Tokenize

2. Do not use Float instead of Single. Need to use compiler directive.

3. Corpus array of byte. Use RawByteString. Done, but check.

4. Drop linked lists. So if you later optimize training hard, use
Tok[i], Prev[i], Next[i], Alive[i]. Not do this now.

5. Drop Head or Tail form linked lists.

6. Use nSymbols, except use nVocab in Transform. Done.

7. Add a regex pretokenizer. Nope, not necessary.

Symbolize.

Should I use clean-up symbols in DisplayByteSymbolTable? Yes, doing so.
Lengthen tabs in printouts like most frequent symbols.array[ or symboltable...1] of Type = ();

Add one pair at the model level, outside ParamBlock:
type
  TWModelParams = record
    ParamBlock: array of TParamBlock;
    Embeddings: TWMatrix;
    FinalGamma: TWVector;
    FinalBeta: TWVector;
  end;

Their sizes are:
SetLength(FinalGamma.Value, ModelDim);
SetLength(FinalGamma.Grad, ModelDim);
SetLength(FinalBeta.Value, ModelDim);
SetLength(FinalBeta.Grad, ModelDim);

Transform/Matrix/Utils.

0. Use CPU/GPU or host/device or cblas/cublas nomenclature?

2. Many models reuse the embedding matrix for output projection.
This is called weight tying. WVocab not needed. I am doing it.

3. Put Hidden on the heap; make it a dynamically allocated variable. No. cblas will not work.

4. Can I make Embeddings a dynamic matrix, and therefore avoid the need for MaxSymbols?
And generally simplify things? It would be the only dynamic variable. No, CBLAS will not work.
It will not work because Embeddings declared as array of array of single, and that creates
a jagged matrix, which is not contiguous. It would work if I make Emebeddings one dimesional.
Also, Probs and TopGradient rely on Embeddings (and Dim Vocab).

5. Use Welford addition. No, not with sgemm.


