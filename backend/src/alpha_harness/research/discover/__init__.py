"""Quant Alpha Idea Discovery: find public research, read what may be read, extract testable
hypotheses and file them in the same library the paper pipeline fills.

* :mod:`.registry`: where research is looked for, and how each place behaved.
* :mod:`.queries`: a broad topic turned into focused, editable research queries.
* :mod:`.robots`, :mod:`.htmltext`: reading a page only where robots.txt and
  the publisher allow, as text, never executing anything in it.
* :mod:`.webextract`: the model prompt and the checks that tie every idea to the page's text.
* :mod:`.dedupe`, :mod:`.ranking`: recognising the same idea twice, and five separate
  indicators that are never added up.
* :mod:`.service`: bounded, persisted jobs; the cache; the request budget.

Nothing here reaches BRAIN or starts a simulation.
"""
