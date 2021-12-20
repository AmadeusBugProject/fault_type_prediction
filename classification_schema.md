
# Semantic

* **"Exception Handling"** faults that are due incorrect exception handling: either missing exception handling or handling an exception improperly that puts the system in a bad state leading to a failure. This fault type is closely related to the next two types, however, if the fault concerns exceptions, this category should be used.
    * "Exception should not be thrown"
    * "Exception should be thrown"
        * https://github.com/netty/netty/issues/8220 
    * "Missing exception handling"
        * https://github.com/elastic/elasticsearch/issues/7086 
    * "Catching too much"
    * "Wrong exception"
        * https://github.com/elastic/elasticsearch/issues/3513 
    * "Exception processing" when the way an exception is handled is incorrect.
        * https://github.com/elastic/elasticsearch/issues/1457 
* **"Missing case"** are faults due to unawareness of a certain case or simply forgotten implementation. 
    * "Null check"
        * https://github.com/square/leakcanary/issues/417 
    * "Missing implementation (Big)
        * https://github.com/elastic/elasticsearch/issues/20912
    * "If (branch) missing"
        * https://github.com/elastic/elasticsearch/issues/850 
    * "If condition missing"
        * https://github.com/bazelbuild/bazel/issues/2474 
    * "Missing case complex"
        * https://github.com/elastic/elasticsearch/issues/11019 
    * "Missing evaluation" when an evaluation or change in variable is missing. (eg. bracketing for a max value before continuing)
    * "Missing case initialization" when the obviously missed case can be resolved by initialization of something, eg. adding a value representing the case to an array.
        * https://github.com/hazelcast/hazelcast/issues/14701 
    * "Call missing" if a method call, or call to callback, listener, etc. is simply missing.
        * https://github.com/elastic/elasticsearch/issues/3381 
* **"Processing"** faults: Something was implemented, but the implementation is incorrect. This can range from simple miscalculations, to incorrect output of a method, to wrong method usage, to wrong library usage, that puts the system in a bad state leading to a failure.
    * "Runaway recursion"
        * https://github.com/libgdx/libgdx/issues/879 
    * "Native overflow" Integer, float, etc overflows. (If the int that is overflowing is used as index for accessing an array it should be considered in "memory" category.
        * https://github.com/haraldk/TwelveMonkeys/issues/514 
    * "Infinite loop" Stuck loops other than recursions.
        * https://github.com/netty/netty/issues/7877 
    * "Method (override) missing" Eg. When a subclass behaves incorrect because the method from superclass is not doing what is supposed to happen in case of this subclass, or a constructor is missing.
        * https://github.com/redisson/redisson/issues/2803 
    * "Design change (Big)" If there are multiple new classes implemented, signatures and apis change, or design overall is adapted.
    * "Complex change" If the design per se stays while internals of classes undergo major changes.
        * https://github.com/elastic/elasticsearch/issues/1032 
    * "Wrong method used"
        * https://github.com/bazelbuild/bazel/issues/9995 
    * "Wrong class used"
        * https://github.com/elastic/elasticsearch/issues/5817 
    * "Wrong datatype"
        * https://github.com/netty/netty/issues/1023 
    * "Re-implementation" Design stays the same while the internals are re implemented. (eg. implementation using regex is re-implemented to custom parsing implementation) (Runaway recursions have their own subsubcategory)
    * "Wrong evaluation" Wrong calculation, evaluation, etc.
        * https://github.com/elastic/elasticsearch/issues/1455 
    * "If condition incorrect" If the conditional statements of an if condition are bound together with wrong operators, or by wrong evaluation. (if there is an additional condition this belongs to missing case) 
    * "Data missing / not propagated" When data isnt propagated or wrongly propagated, or wrong data propagated. eg. Method signatures changed to propagate data to where it is required.
        * https://github.com/jhy/jsoup/issues/977 
    * "Scope wrong" Scope of a variable, method, or class is incorrect. Eg. a static member variable is changed into a instance member variable)
    * "Call order incorrect" if calls are in wrong order (eg. callbacks and listeners)
    * "Wrong/incorrect algorithm" if the used algorithm is the wrong one, or if the algorithm is incorrect. (eg. "runaway recursion" can also be considered a subset, if the algorithm based on recursion is replaced by another implementation of the same algorithm)
        * https://github.com/jhy/jsoup/issues/966 
    * "Incorrect flow" when program flow is simply incorrect, eg. "return" instead of "continue", etc…
    * "Incorrect value" for simply incorrect (default) values that are corrected by putting the right hardcoded value there.
        * https://github.com/bazelbuild/bazel/issues/5830 
    * "Superfluous code" when eg. a method call or evaluation is to be removed without replacement.
        * https://github.com/Azure/azure-iot-sdk-java/issues/205
* **"Typo"** are simple typographic errors.
* **"Dependency"** faults are introduced by changes in a foreign system so that the software can be build, but behaves unexpected. Examples could be recent changes in API or behaviour of underlying OS or utilized libraries that lead to a failure. Please note that the failure has to be introduced by a change in the external/foreign system. Failures caused by not adequately dealing with behaviour of the original external/foreign system fall into other categories.
* **"Other"**  for all semantic faults that do not fall into any of the aforementioned categories. 

# Resources and Memory

* **"Overflow"**
* **"Null pointer dereference"**
* **"Uninitialized memory read"** (except null pointer dereference)
* **"Leak"**
    * "Memory leak"
        * https://github.com/grpc/grpc-java/issues/4198 
    * "Resource leak" Leaking other types of resources as network, file handles, sockets, etc.
        * https://github.com/brettwooldridge/HikariCP/issues/44 
    * "Thread leak"
        * https://github.com/elastic/elasticsearch/issues/9107 
    * "Disk leak" Filling up disk space.
        * https://github.com/jankotek/mapdb/issues/403 
* **"Dangling pointer"**
* **"Double free"** 
* **"Other"**

# Concurrency

* **"Order violation"** faults are caused by missing or incorrect synchronization, e.g.  an object is dereferenced by thread B before it is initialized by thread A.
    * https://github.com/google/ExoPlayer/issues/6146 
* **"Race Condition"**:  two or more threads access the same resource with at least one being a write access and the access is not ordered properly.
    * https://github.com/netty/netty/issues/4829 
* **"Atomic violation"** faults result from lack of constraints on the interleaving of operations. This happens when atomicity of a certain code region was assumed but failed to guarantee atomicity of in the implementation. Please classify a bug only as atomic violation, when it is not an order violation or race condition.
    * "Missing synchronization" Synchronization is missing on a piece of code, Eg. missing @Synchronized on a method, or similar
        * https://github.com/elastic/elasticsearch/issues/43840 
    * "ConcurrentModificationException" Using improper datatypes for concurrent access, eg. concurrently accessing HashMaps that would result in exceptions in contrast to race conditions.
        * https://github.com/checkstyle/checkstyle/issues/4945 
    * "Not atomic" operation on a datatype that is not thread safe, opening the possibility of race conditions. Eg. incrementing java int in multiple threads.
        * https://github.com/checkstyle/checkstyle/issues/4927 
    * "Unintentional value sharing" Eg. not being aware that a variable is static and therefore the value is shared on all threads.
* **"Deadlock"** occurs when two or more threads wait for the other one to release a resource.
    * https://github.com/elastic/elasticsearch/issues/4334 
* **"Parallelisation"** if something is slowing/lagging a thread, and the fix is to have it run by a more appropriate executor, or vice versa, if something multithreaded has to run in a single thread due to some constraints.
    * https://github.com/Automattic/simplenote-android/issues/109 
* **"Other"**
    * "Lock not released" Contrary to a deadlock, there’s no two threads deadlocking on each other, but a method or resource cannot be acquired anymore because the lock was not released on exit. Eg. exception handling within a locked code block without a finally statement leaving the lock acquired while the thread was thrown out of the code block.
        * https://github.com/eclipse/hono/issues/231 

# Other

* **"Documentation"**  for faults or omissions in documentation.
    * https://github.com/google/ExoPlayer/issues/4439 
* **"Build system"** for faults in the build configuration.
    * https://github.com/spring-projects/spring-session/issues/952
* **"Misconfiguration"** for faults in config files. (eg. Dockerfiles, or project internal default config files)
    * https://github.com/eclipse/che/issues/3276    
* **"Ui resources"** for faults due to missing or incorrect ui resource files.
    * https://github.com/codenameone/CodenameOne/issues/1160 
* **"Other"** for any other fault.

