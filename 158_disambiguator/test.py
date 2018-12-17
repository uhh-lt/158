from jsonrpcclient import request


# result = request("http://localhost:5000/", "ping", "test", "lol").data.result
result = request("http://localhost:5002/", "disambiguate", "en", "lol", "kek").data.result
print(result)