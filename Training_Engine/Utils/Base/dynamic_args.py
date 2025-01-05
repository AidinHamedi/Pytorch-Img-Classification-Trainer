# Libs >>>
import time
import json
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
        Creates a hash of env_args for caching purposes.

        Returns:
            str: Hash string of the env_args.

        Example:
            env_args = {'a': 1, 'b': 2}
            hash = self._hash_env_args()  # Generates a consistent hash
        """
        env_str = json.dumps(self.env_args, sort_keys=True)
        return hashlib.sha256(env_str.encode()).hexdigest()

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
