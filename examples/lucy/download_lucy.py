import gdown

url = 'https://drive.google.com/file/d/1ylM3M40leD3hYmg4g65oWbB_dtge88NX/view?usp=sharing'

output = 'lucy.ply'

gdown.download(url, output, quiet=False)
