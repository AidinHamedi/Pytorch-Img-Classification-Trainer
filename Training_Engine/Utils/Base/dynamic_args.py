# You might ask why the F*** does this exist? Because I was having fun using the new deepseek 3 model 😁
# Libs >>>
import time
import pickle
import hashlib
import logging
from typing import Callable, Any, Dict, Optional


# Main >>>
class DynamicArg:
    def __init__(
        self,
        callable: Optional[Callable] = None,
        default_value: Any = None,
        mode: str = "dynamic",
        update_interval: float = 0.0,
        env_args: Optional[Dict] = None,
        cache_enabled: bool = True,
    ):
        """
        Initializes the DynamicArg instance.

        Args:
            callable (Callable, optional): Function to compute the value. Defaults to None.
            default_value (Any, optional): Default value if callable is not provided. Defaults to None.
            mode (str, optional): Operation mode, either 'dynamic' or 'static'. Defaults to 'dynamic'.
            update_interval (float, optional): Time interval for auto-updating in dynamic mode. Defaults to 0.0.
            env_args (Dict, optional): Environment arguments passed to the callable. Defaults to None.
            cache_enabled (bool, optional): Enable caching of callable results based on env_args. Defaults to True.

        Raises:
            ValueError: If mode is not 'dynamic' or 'static'.

        Example:
        ```
            da = DynamicArg(callable=lambda a, b: a + b, default_value=0, mode='dynamic', update_interval=5, env_args={'a': 1, 'b': 2}, cache_enabled=True)
            value = da.get_value()  # Initial call, invokes callable
            time.sleep(6)
            value = da.get_value()  # Calls callable again due to update_interval
        ```
        """
        # Initialize instance variables
        self.callable = callable
        self.default_value = default_value
        self.mode = mode.lower()
        if update_interval < 0:
            self.update_interval = 0.0
        else:
            self.update_interval = update_interval
        self.env_args = env_args if env_args is not None else {}
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.last_env_hash = None
        self.last_updated = time.monotonic()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)

        if self.mode not in ["dynamic", "static"]:
            raise ValueError("Mode must be 'dynamic' or 'static'.")

        if self.mode == "static":
            if callable:
                self.value = self.callable(**self.env_args)
            else:
                self.value = self.default_value
        else:
            self.value = self.default_value
            if callable and self.update_interval > 0:
                self.update()

    def _hash_env_args(self) -> str:
        """
        Hashes the environment arguments using SHA-256, including all serializable parts
        and ignoring non-serializable ones.
        """

        def recursive_hash(obj, hash_obj):
            """
            Recursively hashes parts of the object that can be serialized.
            """
            if isinstance(obj, (int, float, str, bool, bytes)):
                # Basic types are always serializable
                hash_obj.update(pickle.dumps(obj))
            elif isinstance(obj, (list, tuple)):
                # Recursively hash lists and tuples
                for item in obj:
                    recursive_hash(item, hash_obj)
            elif isinstance(obj, dict):
                # Recursively hash dictionaries
                for key, value in sorted(obj.items()):
                    recursive_hash(key, hash_obj)
                    recursive_hash(value, hash_obj)
            elif isinstance(obj, (set, frozenset)):
                # Recursively hash sets
                for item in sorted(obj):
                    recursive_hash(item, hash_obj)
            else:
                try:
                    # Attempt to hash the object
                    hash_obj.update(pickle.dumps(obj))
                except (pickle.PicklingError, AttributeError, TypeError):
                    # Skip non-serializable objects
                    pass

        # Initialize a SHA-256 hash object
        hash_obj = hashlib.sha256()

        # Recursively hash the environment arguments
        recursive_hash(self.env_args, hash_obj)

        # Return the final hash as a hexadecimal string
        return hash_obj.hexdigest()

    def update(self):
        """
        Updates the value by calling the callable with current env_args.

        Handles caching based on cache_enabled flag and update_interval.

        Example:
            self.update()  # Updates value and caches it if conditions are met
        """
        if self.callable is None:
            self.logger.warning("Callable is None, returning default value.")
            self.value = self.default_value
            return
        env_hash = self._hash_env_args()
        if self.cache_enabled and self.update_interval > 0:
            if env_hash in self.cache:
                if time.monotonic() - self.last_updated < self.update_interval:
                    self.value = self.cache[env_hash]
                    return
        try:
            self.value = self.callable(**self.env_args)
            if self.cache_enabled and self.update_interval > 0:
                self.cache[env_hash] = self.value
        except Exception as e:
            self.logger.error(f"Error updating value: {e}")
        self.last_env_hash = env_hash
        self.last_updated = time.monotonic()

    def get_value(self) -> Any:
        """
        Returns the current value based on the mode.

        In 'dynamic' mode, updates the value if necessary based on update_interval or changes in env_args.

        Returns:
            Any: The current value.

        Example:
            value = self.get_value()  # Retrieves the value, updating if necessary
        """
        current_env_hash = self._hash_env_args()
        if self.mode == "dynamic":
            if self.callable is None:
                return self.default_value
            if (
                self.update_interval <= 0
                or time.monotonic() - self.last_updated >= self.update_interval
                or current_env_hash != self.last_env_hash
            ):
                self.update()
            return self.value
        else:
            return self.value

    def set_env_args(self, new_env_args: Dict):
        """
        Sets new environment arguments.

        Triggers an update in 'dynamic' mode if update_interval is positive and env_args have changed.

        Args:
            new_env_args (Dict): New environment arguments.

        Example:
            self.set_env_args({'a': 2, 'b': 3})  # Updates env_args and may trigger an update
        """
        self.env_args = new_env_args
        if self.mode == "dynamic" and self.update_interval > 0:
            current_env_hash = self._hash_env_args()
            if current_env_hash != self.last_env_hash:
                self.update()

    def force_update(self):
        """
        Manually triggers an update of the value, clearing the cache.

        Example:
            self.force_update()  # Forces an immediate update and clears cache
        """
        self.cache = {}
        self.update()

    def reset(self):
        """
        Resets the instance to its initial state.

        Clears cache and resets value to default.

        Example:
            self.reset()  # Resets to initial state
        """
        self.env_args = {}
        if self.callable and self.update_interval > 0:
            self.update()
        else:
            self.value = self.default_value
        self.last_updated = time.monotonic()
        self.cache = {}

    def __str__(self):
        """
        String representation of the DynamicArg instance.

        Returns:
            str: String representation.

        Example:
            print(self)  # Outputs: DynamicArg(value=3, mode='dynamic', last_updated=1672543200.0)
        """
        return f"DynamicArg(value={self.value}, mode={self.mode}, last_updated={self.last_updated if self.mode == 'dynamic' else 'N/A'})"

class DA_Manager:
    """
    A manager class for handling DynamicArg instances and their environment arguments.

    This class is responsible for managing environment arguments (`env_args`) and providing
    a method to retrieve values from DynamicArg instances. It ensures that DynamicArg instances
    are updated with the correct environment arguments and allows for manual updates if needed.

    Attributes:
        env_args (dict): A dictionary of environment arguments that can be passed to DynamicArg instances.
    """

    def __init__(self):
        """
        Initializes the DA_Manager instance.

        Sets up an empty dictionary to store environment arguments.
        """
        self.env_args = {}

    def auto_get(self, dynamic_arg: DynamicArg, manual_update: bool = False) -> Any:
        """
        Retrieves the value from a DynamicArg instance, updating it if necessary.

        This method ensures that the DynamicArg instance is updated with the current environment
        arguments (`env_args`) and retrieves its value. If `manual_update` is True, it forces an
        update of the DynamicArg instance before retrieving the value.

        Args:
            dynamic_arg (DynamicArg): The DynamicArg instance from which to retrieve the value.
            manual_update (bool, optional): If True, forces an update of the DynamicArg instance
                                            before retrieving the value. Defaults to False.

        Returns:
            Any: The current value of the DynamicArg instance.

        Raises:
            ValueError: If the input is not a DynamicArg instance.

        Example:
            ```
            manager = DA_Manager()
            dynamic_arg = DynamicArg(callable=lambda a, b: a + b, default_value=0, mode='dynamic')
            manager.update({'a': 1, 'b': 2})
            value = manager.auto_get(dynamic_arg)  # Retrieves the value, updating if necessary
            ```
        """
        # Check if the input is a DynamicArg instance
        if not isinstance(dynamic_arg, DynamicArg):
            raise ValueError("Input must be a DynamicArg instance.")

        # If the DynamicArg is in dynamic mode, set the environment arguments and update if necessary
        if dynamic_arg.mode == 'dynamic':
            dynamic_arg.set_env_args(self.env_args)
            if manual_update:
                dynamic_arg.update()
            return dynamic_arg.get_value()
        else:
            # In static mode, return the default value
            return dynamic_arg.default_value

    def update(self, env_args: dict):
        """
        Updates the environment arguments managed by the DA_Manager.

        This method sets the `env_args` attribute to the provided dictionary. It is used to
        update the environment arguments that will be passed to DynamicArg instances.

        Args:
            env_args (dict): A dictionary of environment arguments to be used by DynamicArg instances.

        Example:
            ```
            manager = DA_Manager()
            manager.update({'a': 1, 'b': 2})  # Updates the environment arguments
            ```
        """
        self.env_args = env_args