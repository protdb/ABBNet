from Bio.PDB import PDBParser


def get_chain_id(path):
    parser = PDBParser(QUIET=True)
    chain = parser.get_structure('struct', path).get_chains().__next__().id
    print(chain)
    return chain
