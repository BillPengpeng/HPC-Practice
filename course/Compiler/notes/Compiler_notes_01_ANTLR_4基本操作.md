
## 1. Getting Started with ANTLR v4

[Getting Started with ANTLR v4](https://github.com/antlr/antlr4/blob/master/doc/getting-started.md)
[antlr online](http://lab.antlr.org/)

### 1.1 antlr4-tools使用

#### 安装

```bash
$ pip install antlr4-tools

$ antlr4 
Downloading antlr4-4.13.2-complete.jar
ANTLR tool needs Java to run; install Java JRE 11 yes/no (default yes)? y
Installed Java in /Users/parrt/.jre/jdk-11.0.15+10-jre; remove that dir to uninstall
ANTLR Parser Generator  Version 4.13.2
 -o ___              specify output directory where all output is generated
 -lib ___            specify location of grammars, tokens files
...
```

#### 简单示例Expr.g4

``` 
grammar Expr;		
prog:	expr EOF ;
expr:	expr ('*'|'/') expr
    |	expr ('+'|'-') expr
    |	INT
    |	'(' expr ')'
    ;
NEWLINE : [\r\n]+ -> skip;
INT     : [0-9]+ ;
```
(Note: ^D means control-D and indicates "end of input" on Unix; use ^Z on Windows.)

#### Try parsing with a sample grammar

```bash
// To parse and get the parse tree in text form
$ antlr4-parse Expr.g4 prog -tree
10+20*30
^D
(prog:1 (expr:2 (expr:3 10) + (expr:1 (expr:3 20) * (expr:3 30))) <EOF>)

// Here's how to get the tokens and trace through the parse
$ antlr4-parse Expr.g4 prog -tokens -trace
10+20*30
^D

// Here's how to get a visual tree view
$ antlr4-parse Expr.g4 prog -gui
10+20*30
^D
```