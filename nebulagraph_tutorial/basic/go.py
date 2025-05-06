# %%
from nebula_graph import NebulaGraph

ng = NebulaGraph()
# %%
ng.execute(
    """
    GO FROM "player101" 
    OVER follow 
    YIELD id($$) as id, properties($$).name as name;
    """
)
# %%
ng.execute(
    """
    GO FROM "player101" 
    OVER serve
    WHERE serve.start_year > 2010
    YIELD id($$) as id, properties($$).name as name;
    """
)

# %%
