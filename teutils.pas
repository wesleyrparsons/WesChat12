unit TEUtils

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellouth.net, www.wespar.com.}

interface

uses
  Display,
  Global,
  Math,
  SysUtils;

var
  Embeddings: array of array of Single;     // Row is token, column is weights.

implementation

