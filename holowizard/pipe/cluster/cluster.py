from abc import ABC, abstractmethod

class Cluster(ABC):
    """
    Abstract base class for managing Dask clusters.
    Subclasses must implement the lifecycle and scaling methods.
    """

    @abstractmethod
    def create_cluster(self) -> None:
        """
        Configure and initialize the Dask cluster.
        Should be implemented by subclasses to set up the cluster.
        """
        pass

    @abstractmethod
    def start_cluster(self) -> None:
        """
        Start the Dask cluster and connect the client.
        Should be implemented by subclasses to handle the startup process.
        """
        pass

    @abstractmethod
    def stop_cluster(self) -> None:
        """
        Stop the Dask cluster and clean up resources.
        Should be implemented by subclasses to handle the shutdown process.
        """
        pass

    @abstractmethod
    def maintain_workers(self) -> None:
        """
        Keep the number of active workers at the configured target.
        Should be implemented by subclasses to manage worker scaling.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release Dask resources and stop background threads.
        Should be implemented by subclasses to handle cleanup tasks.
        """
        pass
    
    @abstractmethod
    def queue_info(self):
        """
        Retrieve information about the SLURM queue.
        
        Returns:
            dict: Information about the SLURM queue.
        """
        pass