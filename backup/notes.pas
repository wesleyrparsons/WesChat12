unit Notes;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

General
1. Replace nStmbols with nVocab.

2. Where do I use scale, invfreq? Need?

Init state, init grads and probs

3. Test SaveModel and LoadModel procedures.

4. In main program: Read Corpus, Read Files (vocab and merge), Tokenize, Embed, Transform.
One proc: display merge/token info. One proc: display transform/embed info.

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

1. Add a max-heap helps for “what is the most frequent pair right now?”
Instead of scanning all pairs every iteration, keep a heap ordered by count.
But because counts change after merges, you usually do lazy heap updates:
push updated (pair, count, version) records
when popping, discard stale entries. What heap unit to use in FPC?

2. Where does nCorpus live?

3. Do not use Float instead of Single. Need to use compiler directive.

4. Corpus array of byte. Use RawByteString. Done, but check.

5. Drop linked lists. So if you later optimize training hard, use
Tok[i], Prev[i], Next[i], Alive[i].

6. Avoid repeated trie rebuilds.
If the symbol table is fixed, build the trie once after loading. ??

7. Use nSymbols, except use nVocab in Transform. Done.

8. Add a regex pretokenizer. Nope, not necessary.

Symbolize.

Should I use clean-up symbols in DisplayByteSymbolTable? Yes, doing so.
Lengthen tabs in printouts like most frequent symbols.array[ or symboltable...1] of Type = ();

Embed.

The name RunEmbed understates what it does. It seems to:
initialize embeddings, initialize transformer, create training windows,
build input and targets, and run transformer blocks.
No, keep it as Embed.

Transform/Matrix/Utils.

a. Is InvFreq dimmed as ModelDim or HeadDim?

0. ApplyRope inside each rather than across all model.

b. Cuda Softmax only works if SeqLen <= 256.
LaunchSoftMaxForwardN only works in nVocab <= 256.

1. Many models reuse the embedding matrix for output projection.
This is called weight tying. WVocab not needed. I am doing it.

2. What to do with nTokens and append proc.
Store attention softmax outputs. Do I need them intact for backprop through softmax.

3. Put Hidden on the heap; make it a dynamically allocated variable. No. cblas will not work.

4. Can I make Embeddings a dynamic matrix, and therefore avoid the need for MaxSymbols?
And generally simplify things? It would be the only dynamic variable. No, CBLAS will not work.
It will not work because Embeddings declared as array of array of single, and that creates
a jagged matrix, which is not contiguous. It would work if I make Emebeddings one dimesional.
Also, Probs and TopGradient rely on Embeddings (and Dim Vocab).

5. Use Welford addition. No, not with sgemm.


