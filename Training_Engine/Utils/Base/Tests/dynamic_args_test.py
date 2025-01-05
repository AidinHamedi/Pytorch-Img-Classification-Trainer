# Libs >>>
import unittest
from unittest.mock import MagicMock
from ..dynamic_args import DynamicArg
import time


# Modules >>>
def test_callable(**args):
    """A simple callable for testing purposes."""
    return args.get("a", 0) + args.get("b", 0)


class TestDynamicArg(unittest.TestCase):
    def test_cache_disabled(self):
        """Test caching behavior when cache is disabled."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=0,
            env_args={"a": 1},
            cache_enabled=False,
        )
        da.get_value()  # Initial call
        da.get_value()  # Should call again
        da.set_env_args({"a": 1})  # Should not trigger an update immediately
        da.get_value()  # Should call again
        self.assertEqual(callable_mock.call_count, 3)

    def test_cache_enabled(self):
        """Test caching behavior when cache is enabled."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=5,
            env_args={"a": 1},
            cache_enabled=True,
        )
        da.get_value()  # Initial call
        da.set_env_args({"a": 1})  # Should use cache
        da.set_env_args({"a": 2})  # Should call again
        self.assertEqual(callable_mock.call_count, 2)

    def test_cache_reset(self):
        """Test that resetting the instance clears the cache and resets the value."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=5,
            env_args={"a": 1},
            cache_enabled=True,
        )
        da.get_value()  # Initial call
        da.reset()
        self.assertEqual(
            da.get_value(), 100
        )  # Should return callable result after reset
        self.assertEqual(da.cache, {})
        self.assertEqual(callable_mock.call_count, 2)

    def test_negative_update_interval(self):
        """Test behavior with a negative update_interval."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=-1,
            env_args={"a": 1},
        )
        da.get_value()  # Initial call
        da.get_value()  # Should call again
        self.assertEqual(callable_mock.call_count, 2)

    def test_large_update_interval(self):
        """Test behavior with a large update_interval."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=1000,
            env_args={"a": 1},
        )
        da.get_value()  # Initial call
        time.sleep(1)
        da.get_value()  # Should not call again
        self.assertEqual(callable_mock.call_count, 1)

    def test_invalid_env_args(self):
        """Test behavior with invalid env_args."""
        da = DynamicArg(
            callable=test_callable, default_value=0, mode="dynamic", env_args=None
        )
        self.assertEqual(da.get_value(), 0)

    def test_callable_raises_exception(self):
        """Test that exceptions in the callable are handled gracefully."""

        def faulty_callable(**args):
            raise ValueError("Something went wrong.")

        da = DynamicArg(callable=faulty_callable, default_value=0, mode="dynamic")
        self.assertEqual(da.get_value(), 0)

    def test_multiple_env_args_updates(self):
        """Test behavior when env_args are updated multiple times in quick succession."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=0,
            env_args={"a": 1},
        )
        da.get_value()  # Initial call
        da.set_env_args({"a": 2})
        da.get_value()  # Should call again
        da.set_env_args({"a": 3})
        da.get_value()  # Should call again
        self.assertEqual(callable_mock.call_count, 3)

    def test_static_mode_with_callable(self):
        """Test static mode with a callable function."""
        da = DynamicArg(
            callable=test_callable,
            default_value=0,
            mode="static",
            env_args={"a": 1, "b": 2},
        )
        self.assertEqual(da.get_value(), 3)
        da.set_env_args({"a": 2, "b": 3})
        self.assertEqual(da.get_value(), 3)

    def test_static_mode_without_callable(self):
        """Test static mode without a callable function."""
        da = DynamicArg(default_value=42, mode="static")
        self.assertEqual(da.get_value(), 42)
        da.set_env_args({"a": 100})
        self.assertEqual(da.get_value(), 42)

    def test_dynamic_mode_without_callable(self):
        """Test dynamic mode without a callable function."""
        da = DynamicArg(default_value=42, mode="dynamic")
        self.assertEqual(da.get_value(), 42)
        da.set_env_args({"a": 100})
        self.assertEqual(da.get_value(), 42)

    def test_force_update(self):
        """Test that force_update() updates the value immediately."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=10,
            env_args={"a": 1},
        )
        da.get_value()  # Initial call
        da.force_update()  # Should call again
        self.assertEqual(callable_mock.call_count, 2)

    def test_update_interval_zero(self):
        """Test that update_interval=0 forces an update on every call."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=0,
            env_args={"a": 1},
        )
        da.get_value()  # Initial call
        da.get_value()  # Should call again
        self.assertEqual(callable_mock.call_count, 2)

    def test_update_interval_non_zero(self):
        """Test that update_interval > 0 respects the interval."""
        callable_mock = MagicMock(return_value=100)
        da = DynamicArg(
            callable=callable_mock,
            default_value=0,
            mode="dynamic",
            update_interval=2,
            env_args={"a": 1},
            cache_enabled=True,
        )
        da.get_value()  # Initial call
        time.sleep(3)  # Sleep for longer than update_interval
        da.get_value()  # Should call again
        self.assertEqual(callable_mock.call_count, 2)

    def test_invalid_mode(self):
        """Test that an invalid mode raises a ValueError."""
        with self.assertRaises(ValueError):
            DynamicArg(mode="invalid")


if __name__ == "__main__":
    unittest.main()
