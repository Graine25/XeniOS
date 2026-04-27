package jp.xenios;

/**
 * Base class for all unchecked exceptions thrown by the XeniOS project components.
 */
public class XeniosRuntimeException extends RuntimeException {
    public XeniosRuntimeException() {
    }

    public XeniosRuntimeException(final String name) {
        super(name);
    }

    public XeniosRuntimeException(final String name, final Throwable cause) {
        super(name, cause);
    }

    public XeniosRuntimeException(final Exception cause) {
        super(cause);
    }
}
