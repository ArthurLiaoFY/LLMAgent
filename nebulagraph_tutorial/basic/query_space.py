# %%
from nebula_graph import NebulaGraph

ng = NebulaGraph()
# %%
ng.execute(
    """
    MATCH ()<-[e:follow]-() 
    RETURN e 
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH ()<-[e:serve]-() 
    RETURN e 
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH ()<-[e:serve]-() 
    RETURN e 
    LIMIT 3;
    """
)
# %%
ng.execute(
    """
    MATCH (v:player)
    WHERE v.player.name == "Tim Duncan"
    RETURN v
    """
)
# %%
ng.execute(
    """
    MATCH (v:player{name:"Tim Duncan"})-[e:follow{degree:90}]->(v2)
    RETURN e
    """
)
# %%

ng.execute("MATCH (v) RETURN v")


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
