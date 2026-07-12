(function (root) {
  'use strict';

  const STATIC_HOST_SUFFIXES = ['.github.io'];
  const STATIC_RESPONSE_STATUSES = new Set([404, 405, 501]);

  function isStaticHostingLocation(locationLike) {
    const protocol = String(locationLike && locationLike.protocol || '').toLowerCase();
    const hostname = String(locationLike && locationLike.hostname || '').toLowerCase();
    return protocol === 'file:' || STATIC_HOST_SUFFIXES.some(suffix => hostname.endsWith(suffix));
  }

  function responseIndicatesMissingBackend(status, contentType) {
    if (STATIC_RESPONSE_STATUSES.has(Number(status))) return true;
    return !String(contentType || '').toLowerCase().includes('json');
  }

  const api = {
    isStaticHostingLocation,
    responseIndicatesMissingBackend,
  };
  root.PerspectiveRuntime = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
