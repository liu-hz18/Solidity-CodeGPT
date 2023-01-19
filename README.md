# sc-comment

## goals: code-comment inconsistency 

- code-comment bugs
- bad comment [fix comment]

---

## general

- auto-translate comment to sourcecode [semantic code]
  - comment with function signature
  - using seq2seq models and fine-tune with solidity source code corpus
  - corpus consisted: <comment+functionsignature, source code>
  - pre-processing the training set
- comparison between the semantic code with the source code

---

## bug types

- to-from
- to be added

---

## techniques

- runtime execution - no bugs?
- lexical similarity - Jaccard distance
- syntactic similarity - AST edit distance
  - parsing AST https://github.com/solidity-parser/parser