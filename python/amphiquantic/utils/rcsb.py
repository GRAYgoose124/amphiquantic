import requests

def fetch_pdb(pdb_id: str) -> str:
    """ Legacy RCSB API """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    return requests.get(url).text


def save_pdb(pdb_id: str, parent_path: str) -> None:
    """ Legacy RCSB API """
    file_path = f"{parent_path}/{pdb_id}.pdb"
    with open(file_path, "w") as f:
        f.write(fetch_pdb(pdb_id))
