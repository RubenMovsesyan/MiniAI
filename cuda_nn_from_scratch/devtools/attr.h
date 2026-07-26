#ifndef ATTR_H
#define ATTR_H

#define __DEV_ONLY__                                                                                                                                 \
    __attribute__((__warning__(                                                                                                                      \
        "DEVELOPER ONLY: Do not use this in the final "                                                                                              \
        "code!"                                                                                                                                      \
    )))

#define __NOT_YET_IMPLEMENTED__ __attribute__((__error__("Function is not yet implemented, DO NOT USE THIS")))

#endif // ATTR_H
