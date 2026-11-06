"""项目测试包。

存在此文件是为了让标准的测试发现命令可用：

    python -m unittest discover -s tests -t .

没有它时，unittest 会把 tests/ 视为不可导入的目录并报
`ImportError: Start directory is not importable`，
只能改用 `-t tests` 这类非标准写法。
"""
