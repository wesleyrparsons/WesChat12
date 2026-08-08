unit Notes;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wesparsons.com.}

1.  Systematize nTC.
nRawTokenizedCorpus := Length(TokenizedCorpus);
PadToSeqMultiple(TokenizedCorpus, SeqLen);
nTokenizedCorpus := Length(TokenizedCorpus);


3. Replace nSymbols with nVocab.
   Use nTokenizedCorpus instead of Length(TC)

4. Fix GPT2 in infer unit.

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


