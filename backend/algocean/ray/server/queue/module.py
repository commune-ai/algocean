import ray
from ray.util.queue import Queue
from algocean.misc import dict_put,
                     dict_get,
                     dict_has,
                      dict_delete

from algocean import BaseModule
"""

Background Actor for Message Brokers Between Quees

"""

class QueueServer(BaseModule):

    default_cfg_path = 'ray.serve.queue'
    def __init__(self,config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.queue = {}
        
    def delete_topic(self,topic,
                     force=False,
                     grace_period_s=5,
                     verbose=False):

        queue = self.queue.get(topic)
        if isinstance(queue, Queue) :
            queue.shutdown(force=force, grace_period_s=grace_period_s)
            if verbose:
                print(f"{topic} shutdown (force:{force}, grace_period(s): {grace_period_s})")
        else:
            if verbose:
                print(f"{topic} does not exist" )
        # delete queue topic in dict
        self.queue.pop(topic)


    def get_queue(self, topic):
        return self.queue.get(topic)
    
    def topic_exists(self, topic):
        return isinstance(self.queue.get(topic), Queue)



    def create_topic(self, topic,
                     maxsize=10,
                     actor_options=None,
                     refresh=False,
                     verbose=False, **kwargs):


        if self.topic_exists(topic):
            if refresh:
                # kill the queue
                self.delete_topic(topic=topic,force=True)
                queue = Queue(maxsize=maxsize, actor_options=actor_options)

                if verbose:
                    print(f"{topic} Created (maxsize: {maxsize})")

            else:
                if verbose:
                    print(f"{topic} Already Exists (maxsize: {maxsize})")
        else:
            queue = Queue(maxsize=maxsize, actor_options=actor_options)
            if verbose:
                print(f"{topic} Created (maxsize: {maxsize})")

        self.queue[topic] = queue

        return queue 

    def put(self, topic, item, block=True, timeout=None, queue_kwargs=dict(maxsize=10,actor_options=None,refresh=False,verbose=False)):
        """
        Adds an item to the queue.

            If block is True and the queue is full, blocks until the queue is no longer full or until timeout.
            There is no guarantee of order if multiple producers put to the same full queue.
            
            Raises
                Full – if the queue is full and blocking is False.
                
                Full – if the queue is full, blocking is True, and it timed out.
                
                ValueError – if timeout is negative.
        """
        if not self.exists(topic):
            queue_kwargs['topic'] = topic
            self.create_topic(**queue_kwargs)

        self.queue[topic].put(item, block=block, timeout=timeout)


    def get(self, topic, block=True, timeout=None):
        self.queue[topic].get(block=block, timeout=timeout)


    def exists(self, topic):
        return bool(topic in self.topics())

    def delete_all(self):
        for topic in self.topics():
            self.delete_topic(topic, force=True)


    def get_topic(self, topic, *args, **kwargs):
        if dict_has(self.queue, topic):
            return dict_get(self.queue, topic)
        else:
            self.create_topic(topic=topic, **kwargs)
    def size(self, topic):
        # The size of the queue
        return self.get_topic(topic).size()

    def empty(self, topic):
        # Whether the queue is empty.

        return self.get_topic(topic).empty()

    def full(self, topic):
        # Whether the queue is full.
        return self.get_topic(topic).full()


