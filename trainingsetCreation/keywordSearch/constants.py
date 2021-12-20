
# ray: memory leak, null pointer, buffer overflow, heap overflow, null pointer, dangling pointer, double free, segmentation fault
KEYWORDS_MEMORY_RE = ["\sleak",
                     "\soom\s",
                     "allocate",
                     "overflow",
                     "out of memory",
                     "dangling",
                     "dereference",
                     "\sgc\s",
                     "garbage\s*collector",
                     "uninitialized",
                      "\sheap\s",
                      "\sstack\s(?!trace)",
                      "segmentation"]

# ray: deadlock, race condition, synchronization error.
KEYWORDS_CONCURRENCY_RE = ["(:?\s|-)race(:?\s|-|condition)",
                     "deadlock",
                     "\slock",
                     "synchronization",
                     "\sstarvation\s",
                     "\sstarves\s",
                     "\sstarve\s",
                     "(?<!exception in )thread",
                     "atomic",
                     "concurrent",
                     "concurrency"]

# ray programming: exception handling, error handling, type error, typo, compilation error, copy-paste error, refactor- ing, missing switch case, faulty initialization, default value

# ray security: buffer overflow, security, password, oauth, ssl
# ray performance: optimization problem, performance
# ray failure: reboot, crash, hang, restart
KEYWORDS_IMPACT_RE = ["\sanr\s",
                     "\shang\s",
                     "\shangs\s",
                     "\shanging\s",
                     "stuck"]
