import os
from .utils import graphql_query
from algocean import BaseModule
import ray

class GraphQLModule(BaseModule):
    default_cfg_path = "client.graphql.module"

    def __init__(self,config,):
        BaseModule.__init__(self, config)
        self.url = f"http://{config['host']}:{config['port']}"

    def query(self,query, url=None, return_one=False):
        if url != None:
            self.url = url

        
        output = graphql_query(url=self.url, query=query)
        if return_one:
            output = list(output.values())[0]
        
        return output


    def query_list(sef, query_list, num_actors=2, url=None):
        if url != None:
            self.url = url
        
        ray_graphql_query = ray.remote(graphql_query)
        ready_jobs = []
        for query in query_list:
            ready_jobs.append(ray_graphql_query.remote(url=self.url, query=query))
        
        finished_jobs_results  = []
        while ready_jobs:
            ready_jobs, finished_jobs = ray.wait(ready_jobs)
            finished_jobs_results.extend(ray.get(finished_jobs))

        return finished_jobs_results


