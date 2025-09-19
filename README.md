要放到Python原網址https://github.com/acechan0626/pythondata/blob/main/pima-indians-diabetes.csv會出現錯誤
修改辦法
GitHub「網頁」不是「原始 CSV」，所以 pd.read_csv 其實在讀一段 HTML，導致 tokenizer 看到不一致欄位數而報 ParserError。把連結換成 raw 版本就好了。

url ="https://github.com/acechan0626/pythondata/raw/main/pima-indians-diabetes.csv"
